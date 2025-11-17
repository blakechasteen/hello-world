#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Verification Component
============================
Checks consistency and quality of reasoning chains.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation)
"""

import logging
from typing import List

# FIX #4: sklearn now optional in system_id.py (graceful degradation)
from HoloLoom.documentation.types import Context
from HoloLoom.reasoning.types import (
    ReasoningStep,
    VerificationResult,
    VerificationSeverity,
    MultiPassVerification,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Verification Logic
# ============================================================================

class SelfVerifier:
    """
    Verifies consistency and quality of reasoning chains.

    Phase 1: Rule-based verification
    Future: ML-based consistency checking
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize verifier.

        Args:
            threshold: Minimum confidence threshold for passing
        """
        self.threshold = threshold
        self.logger = logging.getLogger(f"{__name__}.SelfVerifier")

    async def verify(
        self,
        chain: List[ReasoningStep],
        context: Context
    ) -> VerificationResult:
        """
        Verify reasoning chain (single pass).

        Args:
            chain: Reasoning chain to verify
            context: Original context

        Returns:
            VerificationResult
        """
        if not chain:
            return VerificationResult(
                passed=False,
                issue="Empty reasoning chain",
                severity=VerificationSeverity.CRITICAL,
                confidence=1.0
            )

        # Check 1: Confidence degradation
        degradation = self._check_confidence_degradation(chain)
        if degradation:
            return degradation

        # Check 2: Evidence consistency
        consistency = self._check_evidence_consistency(chain, context)
        if not consistency.passed:
            return consistency

        # Check 3: Step completeness
        completeness = self._check_completeness(chain)
        if not completeness.passed:
            return completeness

        # All checks passed
        avg_confidence = sum(s.confidence for s in chain) / len(chain)
        return VerificationResult(
            passed=True,
            confidence=avg_confidence
        )

    def _check_confidence_degradation(
        self,
        chain: List[ReasoningStep]
    ) -> VerificationResult:
        """
        Check for severe confidence drops between steps.

        Args:
            chain: Reasoning chain

        Returns:
            VerificationResult if issue found, None otherwise
        """
        for i in range(1, len(chain)):
            prev_conf = chain[i-1].confidence
            curr_conf = chain[i].confidence

            # Severe drop (>0.3) indicates problem
            if prev_conf - curr_conf > 0.3:
                return VerificationResult(
                    passed=False,
                    issue=f"Confidence drop from {prev_conf:.2f} to {curr_conf:.2f} at step {i}",
                    correction="Consider revising reasoning chain to maintain confidence",
                    severity=VerificationSeverity.WARNING,
                    confidence=0.7
                )

        return None

    def _check_evidence_consistency(
        self,
        chain: List[ReasoningStep],
        context: Context
    ) -> VerificationResult:
        """
        Check that evidence is consistent with context.

        Args:
            chain: Reasoning chain
            context: Original context

        Returns:
            VerificationResult
        """
        # For Phase 1, simple check: do we have evidence?
        evidence_steps = [s for s in chain if s.evidence and len(s.evidence) > 10]

        if not evidence_steps and context.shards:
            return VerificationResult(
                passed=False,
                issue="No evidence extracted despite available context",
                correction="Add evidence extraction step",
                severity=VerificationSeverity.WARNING,
                confidence=0.8
            )

        return VerificationResult(passed=True, confidence=0.9)

    def _check_completeness(
        self,
        chain: List[ReasoningStep]
    ) -> VerificationResult:
        """
        Check that chain has all necessary step types.

        Args:
            chain: Reasoning chain

        Returns:
            VerificationResult
        """
        step_types = {s.step_type for s in chain}

        # For STANDARD mode, expect at least understanding + synthesis
        from HoloLoom.reasoning.types import StepType

        required = {StepType.UNDERSTANDING, StepType.SYNTHESIS}
        missing = required - step_types

        if missing:
            return VerificationResult(
                passed=False,
                issue=f"Missing step types: {[t.value for t in missing]}",
                correction="Add missing reasoning steps",
                severity=VerificationSeverity.INFO,
                confidence=0.7
            )

        return VerificationResult(passed=True, confidence=0.95)

    async def verify_multipass(
        self,
        chain: List[ReasoningStep],
        context: Context,
        num_passes: int = 3
    ) -> MultiPassVerification:
        """
        Multi-pass verification for DEEP mode.

        Args:
            chain: Reasoning chain
            context: Original context
            num_passes: Number of verification passes

        Returns:
            MultiPassVerification with all pass results
        """
        passes = []
        contradictions = []

        # Pass 1: Accuracy (confidence checks)
        pass1 = await self.verify(chain, context)
        passes.append(pass1)

        # Pass 2: Completeness (step coverage)
        pass2 = self._verify_completeness_pass(chain)
        passes.append(pass2)

        # Pass 3: Consistency (contradiction detection)
        pass3 = self._verify_consistency_pass(chain, contradictions)
        passes.append(pass3)

        # Calculate overall confidence
        overall_confidence = sum(p.confidence for p in passes) / len(passes)

        return MultiPassVerification(
            passes=passes,
            has_contradictions=len(contradictions) > 0,
            contradictions=contradictions,
            overall_confidence=overall_confidence
        )

    def _verify_completeness_pass(
        self,
        chain: List[ReasoningStep]
    ) -> VerificationResult:
        """
        Completeness verification pass.

        Args:
            chain: Reasoning chain

        Returns:
            VerificationResult for this pass
        """
        # Check minimum length
        if len(chain) < 3:
            return VerificationResult(
                passed=False,
                issue=f"Chain too short ({len(chain)} steps), expected 3+",
                severity=VerificationSeverity.WARNING,
                confidence=0.8
            )

        # Check for reasoning diversity
        step_types = [s.step_type for s in chain]
        unique_types = len(set(step_types))

        if unique_types < 2:
            return VerificationResult(
                passed=False,
                issue="Insufficient reasoning diversity (all same step type)",
                severity=VerificationSeverity.INFO,
                confidence=0.7
            )

        return VerificationResult(passed=True, confidence=0.9)

    def _verify_consistency_pass(
        self,
        chain: List[ReasoningStep],
        contradictions: List[str]
    ) -> VerificationResult:
        """
        Consistency verification pass (contradiction detection).

        Args:
            chain: Reasoning chain
            contradictions: List to populate with found contradictions

        Returns:
            VerificationResult for this pass
        """
        # For Phase 1, simple keyword-based contradiction detection
        # Future: Semantic contradiction detection

        contradiction_keywords = [
            ("always", "never"),
            ("all", "none"),
            ("impossible", "possible"),
            ("true", "false"),
        ]

        thoughts = [s.thought.lower() for s in chain]

        for kw1, kw2 in contradiction_keywords:
            has_kw1 = any(kw1 in t for t in thoughts)
            has_kw2 = any(kw2 in t for t in thoughts)

            if has_kw1 and has_kw2:
                contradictions.append(
                    f"Potential contradiction: chain contains both '{kw1}' and '{kw2}'"
                )

        if contradictions:
            return VerificationResult(
                passed=False,
                issue=f"Found {len(contradictions)} potential contradictions",
                correction="Review chain for logical consistency",
                severity=VerificationSeverity.WARNING,
                confidence=0.6
            )

        return VerificationResult(passed=True, confidence=0.85)
