#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Verifier
================

Verifies that execution achieved the goal.
Integrates with HoloLoom's alignment and reflection systems.

Created: 2025-01-20
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .executor import ExecutionResult
from .planner import ExecutionPlan


@dataclass
class VerificationResult:
    """Result of verification."""
    verified: bool
    confidence: float            # 0.0-1.0
    issues: List[str]            # Problems found
    suggestions: List[str]       # Improvement suggestions
    metadata: Dict[str, Any] = field(default_factory=dict)


class Verifier:
    """
    Verifies execution results against goals.

    Integrates with HoloLoom's alignment framework for self-verification.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def verify(
        self,
        plan: ExecutionPlan,
        execution_result: ExecutionResult
    ) -> VerificationResult:
        """
        Verify execution achieved plan goals.

        Args:
            plan: Original execution plan
            execution_result: Result of code execution

        Returns:
            VerificationResult with confidence and issues
        """
        issues = []
        suggestions = []
        confidence = 1.0

        # Check 1: Execution succeeded
        if not execution_result.success:
            issues.append(f"Execution failed: {execution_result.error}")
            confidence = 0.0
            return VerificationResult(
                verified=False,
                confidence=confidence,
                issues=issues,
                suggestions=["Fix code errors", "Check data format", "Review plan logic"]
            )

        # Check 2: Expected outputs present
        expected_outputs = self._get_expected_outputs(plan)
        missing_outputs = []
        for output in expected_outputs:
            if output not in execution_result.outputs:
                missing_outputs.append(output)

        if missing_outputs:
            issues.append(f"Missing expected outputs: {missing_outputs}")
            confidence *= 0.7
            suggestions.append(f"Ensure code produces: {', '.join(missing_outputs)}")

        # Check 3: Output types reasonable
        type_issues = self._check_output_types(execution_result.outputs)
        if type_issues:
            issues.extend(type_issues)
            confidence *= 0.9

        # Check 4: Query intent satisfied
        intent_satisfied = self._check_intent_satisfied(plan, execution_result)
        if not intent_satisfied:
            issues.append("Query intent may not be fully satisfied")
            confidence *= 0.8
            suggestions.append("Review if results answer the original question")

        # Check 5: No suspicious patterns
        suspicious = self._check_suspicious_patterns(execution_result)
        if suspicious:
            issues.extend(suspicious)
            confidence *= 0.9

        # Determine verification status
        verified = (
            len(issues) == 0 or
            (not self.strict_mode and confidence >= 0.7)
        )

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            metadata={
                'expected_outputs': expected_outputs,
                'actual_outputs': list(execution_result.outputs.keys()),
                'execution_time': execution_result.execution_time
            }
        )

    def _get_expected_outputs(self, plan: ExecutionPlan) -> List[str]:
        """Extract expected output variables from plan."""
        outputs = []
        for step in plan.steps:
            outputs.extend(step.outputs)
        return outputs

    def _check_output_types(self, outputs: Dict[str, Any]) -> List[str]:
        """Check if output types are reasonable."""
        issues = []

        for name, value in outputs.items():
            # Check for empty results
            if isinstance(value, dict):
                if value.get('type') == 'DataFrame':
                    shape = value.get('shape', (0, 0))
                    if shape[0] == 0:
                        issues.append(f"Output '{name}' is an empty DataFrame")

        return issues

    def _check_intent_satisfied(
        self,
        plan: ExecutionPlan,
        execution_result: ExecutionResult
    ) -> bool:
        """Check if original query intent was satisfied."""
        # Basic heuristic: if we have outputs and no errors, likely satisfied
        if not execution_result.outputs:
            return False

        # Check if visualization intent satisfied
        if 'visualize' in plan.metadata.get('intent', {}).get('operations', []):
            # Check if plot was created (plt.show called or figure exists)
            if 'plot' not in execution_result.outputs:
                return False

        # More sophisticated checks would integrate with HoloLoom's semantic understanding
        return True

    def _check_suspicious_patterns(self, execution_result: ExecutionResult) -> List[str]:
        """Check for suspicious patterns in results."""
        suspicious = []

        # Check stderr for warnings
        if execution_result.stderr:
            if 'warning' in execution_result.stderr.lower():
                suspicious.append("Execution produced warnings")

        # Check for very long execution times
        if execution_result.execution_time > 10.0:
            suspicious.append(f"Long execution time: {execution_result.execution_time:.1f}s")

        return suspicious

    def suggest_improvements(
        self,
        plan: ExecutionPlan,
        verification: VerificationResult
    ) -> List[str]:
        """
        Suggest improvements to plan based on verification.

        Args:
            plan: Original plan
            verification: Verification result

        Returns:
            List of improvement suggestions
        """
        improvements = list(verification.suggestions)

        # Add plan-specific suggestions
        if plan.estimated_complexity == "high":
            improvements.append("Consider breaking into smaller sub-queries")

        if verification.confidence < 0.8:
            improvements.append("Low confidence - consider alternative approach")

        # Check for missing error handling
        for step in plan.steps:
            if 'try' not in step.code_template.lower():
                improvements.append(f"Step {step.step_id}: Add error handling")
                break  # Only suggest once

        return improvements


# Convenience function
def verify_execution(
    plan: ExecutionPlan,
    execution_result: ExecutionResult
) -> VerificationResult:
    """
    Quick verification.

    Args:
        plan: Execution plan
        execution_result: Execution result

    Returns:
        VerificationResult
    """
    verifier = Verifier()
    return verifier.verify(plan, execution_result)
