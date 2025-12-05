"""
Fix Strategy Selector
=====================

Selects the appropriate fix strategy based on issue category and context:

- AST: Automated AST transformations (extract function, rename, etc.)
- TEMPLATE: Template-based fixes (add try/except, move to env, etc.)
- MANUAL: Requires human implementation
- SKIP: Don't fix (false positive or too risky)

Strategy selection is based on:
- Issue category (some categories map to specific strategies)
- Risk level (high risk → manual)
- Context (comment/string → skip)
- Code complexity (high complexity → manual)
"""

from typing import List, Optional
from .xterminator_types import FixStrategy, RiskLevel, CodeContext, ContextType


class StrategySelector:
    """Selects appropriate fix strategy"""

    # Category → Primary strategy mapping
    CATEGORY_STRATEGY = {
        # AST transformations
        'copy_paste': FixStrategy.AST,        # Extract common function
        'dead_code': FixStrategy.AST,         # Remove unused code
        'unused_import': FixStrategy.AST,     # Remove import
        'poor_naming': FixStrategy.AST,       # Rename
        'redundant_code': FixStrategy.AST,    # Simplify

        # Template-based
        'error_handling': FixStrategy.TEMPLATE,   # Add try/except
        'hardcoded_values': FixStrategy.TEMPLATE, # Move to config/env
        'resource_management': FixStrategy.TEMPLATE, # Add with statement

        # Manual only
        'security': FixStrategy.MANUAL,
        'sql_injection': FixStrategy.MANUAL,
        'xss': FixStrategy.MANUAL,
        'logic_error': FixStrategy.MANUAL,  # Logic requires understanding
        'state_mutation': FixStrategy.MANUAL,
        'async_issues': FixStrategy.MANUAL,
        'type_confusion': FixStrategy.MANUAL,
    }

    def __init__(self):
        pass

    def select_strategy(
        self,
        issue_category: str,
        risk_level: RiskLevel,
        context: CodeContext,
        confidence: float
    ) -> tuple[FixStrategy, str, List[FixStrategy]]:
        """
        Select fix strategy.

        Args:
            issue_category: Issue category from Trough
            risk_level: Assessed risk level
            context: Code context
            confidence: Detection confidence

        Returns:
            (selected_strategy, reason, alternative_strategies)
        """
        alternatives = []
        reason = ""

        # Rule 1: Skip if not in executable code
        if not context.is_safe_location():
            return (
                FixStrategy.SKIP,
                f"Issue in {context.context_type.value}, not executable code",
                []
            )

        # Rule 2: Skip if confidence is very low (likely false positive)
        if confidence < 0.50:
            return (
                FixStrategy.SKIP,
                f"Low confidence ({confidence:.2f}) - likely false positive",
                []
            )

        # Rule 3: Manual only if CRITICAL risk
        if risk_level == RiskLevel.CRITICAL:
            return (
                FixStrategy.MANUAL,
                "Critical risk level - requires manual review",
                []
            )

        # Rule 4: Manual if HIGH risk and low confidence
        if risk_level == RiskLevel.HIGH and confidence < 0.75:
            return (
                FixStrategy.MANUAL,
                f"High risk with low confidence ({confidence:.2f})",
                []
            )

        # Rule 5: Get category-based strategy
        primary_strategy = self.CATEGORY_STRATEGY.get(
            issue_category,
            FixStrategy.MANUAL  # Default to manual if unknown
        )

        # Rule 6: Downgrade to manual if no tests
        if not context.has_tests and primary_strategy in {FixStrategy.AST, FixStrategy.TEMPLATE}:
            alternatives.append(primary_strategy)
            return (
                FixStrategy.MANUAL,
                f"No test coverage - {primary_strategy.value} fix available but needs manual testing",
                alternatives
            )

        # Rule 7: Consider alternatives based on context
        if primary_strategy == FixStrategy.AST:
            # AST might not work for complex code - template could be alternative
            if context.scope_depth > 3:
                alternatives.append(FixStrategy.TEMPLATE)
                alternatives.append(FixStrategy.MANUAL)
                reason = f"Selected {primary_strategy.value} (deep nesting - consider alternatives)"
            else:
                reason = f"Selected {primary_strategy.value} for automated transformation"

        elif primary_strategy == FixStrategy.TEMPLATE:
            # Template fixes are generally safe
            alternatives.append(FixStrategy.AST)
            alternatives.append(FixStrategy.MANUAL)
            reason = f"Selected {primary_strategy.value} for template-based fix"

        elif primary_strategy == FixStrategy.MANUAL:
            reason = f"Category '{issue_category}' requires manual implementation"

        # Return primary strategy
        return primary_strategy, reason, alternatives

    def can_automate(
        self,
        strategy: FixStrategy,
        risk_level: RiskLevel,
        confidence: float,
        has_tests: bool
    ) -> bool:
        """
        Check if strategy can be automated.

        Criteria:
        - Strategy must be AST or TEMPLATE
        - Risk must be LOW or MEDIUM
        - Confidence must be ≥0.80
        - Must have tests
        """
        if strategy not in {FixStrategy.AST, FixStrategy.TEMPLATE}:
            return False

        if risk_level not in {RiskLevel.LOW, RiskLevel.MEDIUM}:
            return False

        if confidence < 0.80:
            return False

        if not has_tests:
            return False

        return True

    def get_implementation_hints(
        self,
        strategy: FixStrategy,
        issue_category: str,
        context: CodeContext
    ) -> List[str]:
        """Get hints for implementing the fix"""
        hints = []

        if strategy == FixStrategy.AST:
            if issue_category == 'copy_paste':
                hints.append("Extract duplicated code into a common function")
                hints.append("Parameterize differences between copies")
                hints.append("Update all call sites to use new function")

            elif issue_category == 'dead_code':
                hints.append("Remove unreachable code after return/raise")
                hints.append("Consider if code was meant to be conditional")

            elif issue_category == 'unused_import':
                hints.append("Remove unused import statement")
                hints.append("Check if import is used in type annotations")

            elif issue_category == 'poor_naming':
                hints.append("Rename to descriptive name following PEP 8")
                hints.append("Update all references in scope")

        elif strategy == FixStrategy.TEMPLATE:
            if issue_category == 'error_handling':
                hints.append("Wrap risky operation in try/except")
                hints.append("Catch specific exception types, not bare except")
                hints.append("Add logging in except block")
                hints.append("Consider re-raising with more context")

            elif issue_category == 'hardcoded_values':
                hints.append("Move value to module-level constant")
                hints.append("Or read from environment variable")
                hints.append("Or load from config file")
                hints.append("Document expected format/values")

            elif issue_category == 'resource_management':
                hints.append("Use context manager (with statement)")
                hints.append("Ensure resource is closed in finally block")

        elif strategy == FixStrategy.MANUAL:
            hints.append("This issue requires human judgment")
            hints.append("Review code carefully before fixing")
            hints.append("Consider security implications")
            hints.append("Add tests before and after fix")

        # Add context-specific hints
        if context.in_try_block:
            hints.append("Note: Code is already in try/except block")

        if context.in_loop:
            hints.append("Note: Code is inside loop - consider performance")

        if not context.has_tests:
            hints.append("⚠️ No tests detected - add tests before fixing")

        return hints
