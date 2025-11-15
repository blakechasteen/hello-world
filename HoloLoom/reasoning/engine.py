#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine - Layer 6
===========================
Multi-step reasoning layer for explicit chain-of-thought generation.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation - FAST and STANDARD modes)
"""

import asyncio
import logging
import time
from typing import Optional

from HoloLoom.documentation.types import Query, Features, Context
from HoloLoom.reasoning.types import (
    ReasoningMode,
    ReasoningResult,
    ReasoningStep,
    StepType,
    DEFAULT_CONFIDENCE_THRESHOLDS,
)
from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.verifier import SelfVerifier

# Optional scratchpad integration
try:
    from Promptly.promptly.recursive_loops import Scratchpad
    SCRATCHPAD_AVAILABLE = True
except ImportError:
    SCRATCHPAD_AVAILABLE = False
    Scratchpad = None

logger = logging.getLogger(__name__)


# ============================================================================
# Reasoning Engine
# ============================================================================

class ReasoningEngine:
    """
    Multi-step reasoning layer for explicit chain-of-thought generation.

    Modes:
    - FAST: Single-pass reasoning (<50ms overhead)
    - STANDARD: Multi-step CoT (3-5 steps, ~200ms)
    - DEEP: Planning + verification + backtracking (~500ms+) [Phase 3]

    Phase 1: FAST and STANDARD modes only
    """

    def __init__(
        self,
        mode: ReasoningMode = ReasoningMode.STANDARD,
        scratchpad: Optional[Scratchpad] = None,
        max_thinking_steps: int = 5,
        verification_threshold: float = 0.75
    ):
        """
        Initialize reasoning engine.

        Args:
            mode: Default reasoning mode
            scratchpad: Optional scratchpad for provenance
            max_thinking_steps: Maximum reasoning steps
            verification_threshold: Minimum confidence for passing
        """
        self.mode = mode
        self.scratchpad = scratchpad
        self.max_steps = max_thinking_steps
        self.verification_threshold = verification_threshold

        # Components
        self.planner = QueryPlanner()
        self.cot_generator = ChainOfThought()
        self.verifier = SelfVerifier(threshold=verification_threshold)

        self.logger = logging.getLogger(f"{__name__}.ReasoningEngine")

        if not SCRATCHPAD_AVAILABLE and scratchpad:
            self.logger.warning(
                "Scratchpad requested but Promptly not available. "
                "Continuing without scratchpad integration."
            )
            self.scratchpad = None

    async def reason(
        self,
        query: Query,
        features: Features,
        context: Context,
        mode: Optional[ReasoningMode] = None
    ) -> ReasoningResult:
        """
        Generate reasoning chain for query.

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context
            mode: Override default mode

        Returns:
            ReasoningResult with reasoning chain
        """
        start_time = time.time()
        reasoning_mode = mode or self.mode

        # Route to appropriate reasoning method
        if reasoning_mode == ReasoningMode.FAST:
            result = await self.reason_fast(features, context)
        elif reasoning_mode == ReasoningMode.STANDARD:
            result = await self.reason_standard(query, features, context)
        elif reasoning_mode == ReasoningMode.DEEP:
            # Phase 3 - for now, fall back to STANDARD
            self.logger.warning(
                "DEEP mode requested but not yet implemented (Phase 3). "
                "Falling back to STANDARD mode."
            )
            result = await self.reason_standard(query, features, context)
        else:
            raise ValueError(f"Unknown reasoning mode: {reasoning_mode}")

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        result.duration_ms = duration_ms

        self.logger.info(
            f"Reasoning complete: mode={result.mode.value}, "
            f"steps={len(result.chain)}, confidence={result.total_confidence:.2f}, "
            f"duration={duration_ms:.1f}ms"
        )

        return result

    async def reason_fast(
        self,
        features: Features,
        context: Context
    ) -> ReasoningResult:
        """
        Fast reasoning: Quick sanity check before decision.

        Steps:
        1. Estimate confidence based on context quality
        2. If high confidence (>0.85), proceed directly
        3. If low confidence (<0.7), escalate to STANDARD mode

        Args:
            features: Extracted features
            context: Retrieved context

        Returns:
            ReasoningResult with single-step chain or escalation
        """
        # Quick confidence estimation
        confidence = self.cot_generator.estimate_confidence(features, context)

        if confidence >= DEFAULT_CONFIDENCE_THRESHOLDS[ReasoningMode.FAST]:
            # High confidence - direct answer mode
            chain = [
                ReasoningStep(
                    thought="Context directly answers query with high confidence",
                    evidence=f"{len(context.shards)} relevant shards found",
                    confidence=confidence,
                    step_type=StepType.UNDERSTANDING
                )
            ]

            return ReasoningResult(
                chain=chain,
                mode=ReasoningMode.FAST,
                total_confidence=confidence
            )

        # Low confidence - escalate to STANDARD
        self.logger.info(
            f"FAST mode confidence too low ({confidence:.2f}), "
            f"escalating to STANDARD mode"
        )

        # Note: In production, this would recursively call reason_standard
        # For Phase 1, we just return a low-confidence result
        chain = [
            ReasoningStep(
                thought=f"Confidence too low ({confidence:.2f}) for FAST mode",
                evidence="Escalation to STANDARD mode recommended",
                confidence=confidence,
                step_type=StepType.UNDERSTANDING
            )
        ]

        return ReasoningResult(
            chain=chain,
            mode=ReasoningMode.FAST,
            total_confidence=confidence,
            metadata={"escalation_recommended": True}
        )

    async def reason_standard(
        self,
        query: Query,
        features: Features,
        context: Context
    ) -> ReasoningResult:
        """
        Standard reasoning: Multi-step chain-of-thought.

        Steps:
        1. Analyze query intent (what are they really asking?)
        2. Identify key evidence from context
        3. Synthesize reasoning chain
        4. Self-verification check
        5. If verification fails, add corrective step

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context

        Returns:
            ReasoningResult with 3-5 step chain
        """
        # Step 1: Analyze intent
        intent = self.planner.analyze_intent(query, features)

        # Step 2-3: Generate reasoning chain
        chain = self.cot_generator.generate_standard_chain(query, intent, context)

        # Limit to max steps
        if len(chain) > self.max_steps:
            self.logger.warning(
                f"Chain has {len(chain)} steps, truncating to {self.max_steps}"
            )
            chain = chain[:self.max_steps]

        # Step 4: Self-verification
        verification = await self.verifier.verify(chain, context)

        # Step 5: Add corrective step if needed
        if not verification.passed:
            chain.append(ReasoningStep(
                thought=f"Verification issue detected: {verification.issue}",
                evidence=verification.correction or "Manual review recommended",
                confidence=0.5,
                step_type=StepType.CORRECTION
            ))

            self.logger.info(
                f"Verification failed: {verification.issue} "
                f"(severity={verification.severity.value})"
            )

        # Calculate total confidence
        total_confidence = sum(s.confidence for s in chain) / len(chain)

        return ReasoningResult(
            chain=chain,
            mode=ReasoningMode.STANDARD,
            total_confidence=total_confidence,
            metadata={
                "intent": intent.type.value,
                "complexity": intent.complexity,
                "verification_passed": verification.passed
            }
        )

    def estimate_confidence(
        self,
        features: Features,
        context: Context
    ) -> float:
        """
        Quick confidence estimation without full reasoning.

        Args:
            features: Extracted features
            context: Retrieved context

        Returns:
            Estimated confidence [0.0, 1.0]
        """
        return self.cot_generator.estimate_confidence(features, context)

    def select_mode(
        self,
        query: Query,
        features: Features,
        context: Context
    ) -> ReasoningMode:
        """
        Select appropriate reasoning mode based on query characteristics.

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context

        Returns:
            Recommended ReasoningMode
        """
        # Quick confidence check
        confidence = self.estimate_confidence(features, context)

        # Analyze intent for complexity
        intent = self.planner.analyze_intent(query, features)

        # Decision logic
        if confidence >= 0.85 and intent.complexity < 0.4:
            # Simple query, high confidence → FAST
            return ReasoningMode.FAST

        elif confidence < 0.5 or intent.complexity > 0.7:
            # Low confidence or high complexity → DEEP
            # (Phase 3 - for now falls back to STANDARD)
            return ReasoningMode.DEEP

        else:
            # Default to STANDARD
            return ReasoningMode.STANDARD


# ============================================================================
# Convenience Functions
# ============================================================================

async def reason_with_mode(
    query: Query,
    features: Features,
    context: Context,
    mode: ReasoningMode = ReasoningMode.STANDARD,
    **kwargs
) -> ReasoningResult:
    """
    Convenience function for one-off reasoning.

    Args:
        query: Input query
        features: Extracted features
        context: Retrieved context
        mode: Reasoning mode
        **kwargs: Additional engine parameters

    Returns:
        ReasoningResult
    """
    engine = ReasoningEngine(mode=mode, **kwargs)
    return await engine.reason(query, features, context)


async def auto_reason(
    query: Query,
    features: Features,
    context: Context,
    **kwargs
) -> ReasoningResult:
    """
    Automatically select and apply reasoning mode.

    Args:
        query: Input query
        features: Extracted features
        context: Retrieved context
        **kwargs: Additional engine parameters

    Returns:
        ReasoningResult with auto-selected mode
    """
    engine = ReasoningEngine(**kwargs)
    selected_mode = engine.select_mode(query, features, context)

    return await engine.reason(query, features, context, mode=selected_mode)
