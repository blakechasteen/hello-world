#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine - Layer 6 (HARDENED)
======================================
Multi-step reasoning layer for explicit chain-of-thought generation.

PRODUCTION HARDENING:
- ✅ Modular: Components are truly swappable via dependency injection
- ✅ Timeout protected: All operations have configurable timeouts
- ✅ Error boundaries: Graceful degradation on component failures
- ✅ Resource limited: Max chain length, memory bounds
- ✅ Battle tested: Comprehensive error handling

Author: Claude Code
Date: 2025-11-17
Phase: Hardened for production
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any

# Direct imports to avoid sklearn dependency cycle
from HoloLoom.reasoning.types import (
    ReasoningMode,
    ReasoningResult,
    ReasoningStep,
    StepType,
    DEFAULT_CONFIDENCE_THRESHOLDS,
)
from HoloLoom.reasoning.resource_limits import (
    ResourceMonitor,
    ResourceLimits,
    ResourceLimitError,
    ChainLengthLimitError,
)
from HoloLoom.reasoning.metrics import (
    ReasoningMetrics,
    track_reasoning,
)

# Optional scratchpad integration
try:
    from Promptly.promptly.recursive_loops import Scratchpad
    SCRATCHPAD_AVAILABLE = True
except ImportError:
    SCRATCHPAD_AVAILABLE = False
    Scratchpad = None

logger = logging.getLogger(__name__)


# ============================================================================
# Reasoning Engine (HARDENED)
# ============================================================================

class ReasoningEngine:
    """
    Multi-step reasoning layer for explicit chain-of-thought generation.

    PRODUCTION FEATURES:
    - Dependency injection: All components swappable
    - Timeout protection: Configurable per-operation timeouts
    - Error boundaries: Graceful degradation on failures
    - Resource limits: Memory, chain length, concurrency bounds

    Modes:
    - FAST: Single-pass reasoning (<50ms overhead)
    - STANDARD: Multi-step CoT (3-5 steps, ~200ms)
    - DEEP: Planning + verification + backtracking (~500ms+)
    """

    def __init__(
        self,
        mode: ReasoningMode = ReasoningMode.STANDARD,
        reasoner: Optional[Any] = None,
        verifier: Optional[Any] = None,
        planner: Optional[Any] = None,
        backtracker: Optional[Any] = None,
        scratchpad: Optional[Scratchpad] = None,
        max_thinking_steps: int = 5,
        verification_threshold: float = 0.75,
        timeout_ms: float = 500.0,
        enable_fallback: bool = True,
        resource_limits: Optional[ResourceLimits] = None
    ):
        """
        Initialize reasoning engine with dependency injection.

        Args:
            mode: Default reasoning mode
            reasoner: Custom ChainOfThought component (optional)
            verifier: Custom SelfVerifier component (optional)
            planner: Custom QueryPlanner component (optional)
            backtracker: Custom Backtracker component (optional)
            scratchpad: Optional scratchpad for provenance
            max_thinking_steps: Maximum reasoning steps (resource limit)
            verification_threshold: Minimum confidence for passing
            timeout_ms: Timeout for reasoning operations (DoS protection)
            enable_fallback: Enable graceful degradation on errors
            resource_limits: Custom resource limits (optional)
        """
        # Validation
        if not 1 <= max_thinking_steps <= 50:
            raise ValueError("max_thinking_steps must be between 1 and 50")
        if not 0.0 <= verification_threshold <= 1.0:
            raise ValueError("verification_threshold must be between 0.0 and 1.0")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        self.mode = mode
        self.scratchpad = scratchpad
        self.max_steps = max_thinking_steps
        self.verification_threshold = verification_threshold
        self.timeout_ms = timeout_ms
        self.enable_fallback = enable_fallback

        # PHASE 2: Resource monitoring and limits
        if resource_limits is None:
            resource_limits = ResourceLimits(
                max_chain_steps=max_thinking_steps,
                max_memory_mb=512.0,
                max_concurrent_operations=100
            )
        self.resource_monitor = ResourceMonitor(resource_limits)

        # PHASE 2: Metrics tracking
        self.metrics = ReasoningMetrics()

        # Lazy import to avoid circular dependencies
        from HoloLoom.reasoning.planner import QueryPlanner
        from HoloLoom.reasoning.chain_of_thought import ChainOfThought
        from HoloLoom.reasoning.verifier import SelfVerifier
        from HoloLoom.reasoning.backtracker import Backtracker

        # FIX #1: MODULARITY - Components are truly swappable
        self.planner = planner or QueryPlanner()
        self.reasoner = reasoner or ChainOfThought()  # Changed from cot_generator!
        self.verifier = verifier or SelfVerifier(threshold=verification_threshold)
        self.backtracker = backtracker or Backtracker(max_revisions=3)

        self.logger = logging.getLogger(f"{__name__}.ReasoningEngine")

        if not SCRATCHPAD_AVAILABLE and scratchpad:
            self.logger.warning(
                "Scratchpad requested but Promptly not available. "
                "Continuing without scratchpad integration."
            )
            self.scratchpad = None

    async def reason(
        self,
        query: Any,
        features: Any,
        context: Any,
        mode: Optional[ReasoningMode] = None,
        timeout_ms: Optional[float] = None
    ) -> ReasoningResult:
        """
        Generate reasoning chain for query with timeout protection.

        FIX #2: TIMEOUT HANDLING - Wraps with asyncio.wait_for()

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context
            mode: Override default mode
            timeout_ms: Override timeout

        Returns:
            ReasoningResult with reasoning chain

        Raises:
            asyncio.TimeoutError: If reasoning exceeds timeout
        """
        timeout = (timeout_ms or self.timeout_ms) / 1000  # Convert to seconds

        try:
            # PHASE 2: Resource limiting - Track concurrent operations
            async with self.resource_monitor.operation_context():
                # Check memory before starting
                if not self.resource_monitor.check_memory_limit():
                    raise ResourceLimitError("Memory limit exceeded before reasoning")

                # Check context size
                context_size = len(context.shards) if hasattr(context, 'shards') else 0
                if not self.resource_monitor.check_context_size(context_size):
                    raise ResourceLimitError(f"Context too large: {context_size} shards")

                # FIX #2: TIMEOUT - Wrap with asyncio.wait_for()
                result = await asyncio.wait_for(
                    self._reason_impl(query, features, context, mode),
                    timeout=timeout
                )
                return result

        except asyncio.TimeoutError:
            self.logger.error(
                f"Reasoning timeout after {timeout*1000:.1f}ms "
                f"(mode={mode or self.mode})"
            )

            if self.enable_fallback:
                # FIX #3: ERROR BOUNDARY - Return fallback result
                return self._create_timeout_fallback(query, mode or self.mode)
            else:
                raise

    async def _reason_impl(
        self,
        query: Any,
        features: Any,
        context: Any,
        mode: Optional[ReasoningMode] = None
    ) -> ReasoningResult:
        """
        Internal reasoning implementation (wrapped by timeout).

        FIX #3: ERROR BOUNDARIES - All external calls wrapped in try/except
        """
        start_time = time.time()
        reasoning_mode = mode or self.mode

        try:
            # Route to appropriate reasoning method
            if reasoning_mode == ReasoningMode.FAST:
                result = await self.reason_fast(features, context)
            elif reasoning_mode == ReasoningMode.STANDARD:
                result = await self.reason_standard(query, features, context)
            elif reasoning_mode == ReasoningMode.DEEP:
                result = await self.reason_deep(query, features, context)
            else:
                raise ValueError(f"Unknown reasoning mode: {reasoning_mode}")

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms

            # PHASE 2: Track metrics
            self.metrics.track_operation(
                mode=result.mode.value,
                duration_ms=duration_ms,
                confidence=result.total_confidence,
                chain_length=len(result.chain),
                success=True
            )

            self.logger.info(
                f"Reasoning complete: mode={result.mode.value}, "
                f"steps={len(result.chain)}, confidence={result.total_confidence:.2f}, "
                f"duration={duration_ms:.1f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}", exc_info=True)

            # PHASE 2: Track error metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.track_operation(
                mode=reasoning_mode.value,
                duration_ms=duration_ms,
                confidence=0.0,
                chain_length=0,
                success=False
            )

            if self.enable_fallback:
                # FIX #3: ERROR BOUNDARY - Graceful degradation
                return self._create_error_fallback(query, reasoning_mode, str(e))
            else:
                raise

    async def reason_fast(
        self,
        features: Any,
        context: Any
    ) -> ReasoningResult:
        """
        Fast reasoning with error boundaries.

        FIX #3: ERROR BOUNDARIES - Wrapped try/except
        """
        try:
            # Quick confidence estimation
            confidence = self.reasoner.estimate_confidence(features, context)

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

            # Low confidence - return escalation recommendation
            self.logger.info(
                f"FAST mode confidence too low ({confidence:.2f}), "
                f"escalation to STANDARD recommended"
            )

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

        except Exception as e:
            self.logger.error(f"FAST reasoning failed: {e}")

            if self.enable_fallback:
                return self._create_mode_fallback(ReasoningMode.FAST, str(e))
            else:
                raise

    async def reason_standard(
        self,
        query: Any,
        features: Any,
        context: Any
    ) -> ReasoningResult:
        """
        Standard reasoning with error boundaries.

        FIX #3: ERROR BOUNDARIES - Each component call wrapped
        """
        chain = []

        try:
            # Step 1: Analyze intent (ERROR BOUNDARY)
            try:
                intent = self.planner.analyze_intent(query, features)
            except Exception as e:
                self.logger.error(f"Intent analysis failed: {e}")
                # Fallback to simple intent
                from HoloLoom.reasoning.types import QueryIntent, QueryType
                intent = QueryIntent(
                    type=QueryType.FACTUAL,
                    requirements=[],
                    key_concepts=[],
                    complexity=0.5,
                    confidence=0.5
                )

            # Step 2-3: Generate reasoning chain (ERROR BOUNDARY)
            try:
                chain = self.reasoner.generate_standard_chain(query, intent, context)
            except Exception as e:
                self.logger.error(f"Chain generation failed: {e}")
                # Fallback to minimal chain
                chain = [
                    ReasoningStep(
                        thought="Reasoning chain generation failed, using fallback",
                        evidence=str(e),
                        confidence=0.3,
                        step_type=StepType.UNDERSTANDING
                    )
                ]

            # PHASE 2: Check chain length resource limit
            if not self.resource_monitor.check_chain_length(len(chain)):
                self.logger.warning(
                    f"Chain length {len(chain)} exceeds limit, truncating"
                )
                chain = chain[:self.max_steps]

            # Limit to max steps (legacy check for backward compat)
            if len(chain) > self.max_steps:
                self.logger.warning(
                    f"Chain has {len(chain)} steps, truncating to {self.max_steps}"
                )
                chain = chain[:self.max_steps]

            # Step 4: Self-verification (ERROR BOUNDARY)
            try:
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

            except Exception as e:
                self.logger.error(f"Verification failed: {e}")
                # Continue without verification

            # Calculate total confidence
            total_confidence = sum(s.confidence for s in chain) / len(chain) if chain else 0.0

            return ReasoningResult(
                chain=chain,
                mode=ReasoningMode.STANDARD,
                total_confidence=total_confidence,
                metadata={
                    "intent": intent.type.value,
                    "complexity": intent.complexity,
                    "verification_attempted": True
                }
            )

        except Exception as e:
            self.logger.error(f"STANDARD reasoning failed: {e}")

            if self.enable_fallback:
                return self._create_mode_fallback(ReasoningMode.STANDARD, str(e))
            else:
                raise

    async def reason_deep(
        self,
        query: Any,
        features: Any,
        context: Any
    ) -> ReasoningResult:
        """
        Deep reasoning with comprehensive error boundaries.

        FIX #3: ERROR BOUNDARIES - Each phase wrapped independently
        """
        chain = []

        try:
            # Step 1: Planning (ERROR BOUNDARY)
            try:
                self.logger.info("DEEP mode: Creating query plan")
                plan = self.planner.create_plan(query, features, context)
                intent = self.planner.analyze_intent(query, features)

                chain.append(ReasoningStep(
                    thought=f"Query plan created: {len(plan.steps)} sub-questions to answer",
                    evidence="; ".join([s.question[:60] + "..." if len(s.question) > 60 else s.question
                                      for s in plan.steps[:3]]),
                    confidence=0.9,
                    step_type=StepType.PLANNING,
                    metadata={
                        "plan_complexity": plan.estimated_complexity,
                        "num_steps": len(plan.steps)
                    }
                ))

            except Exception as e:
                self.logger.error(f"Planning failed: {e}")
                # Fallback to STANDARD mode
                return await self.reason_standard(query, features, context)

            # Step 2: Execute plan steps (ERROR BOUNDARY per step)
            self.logger.info(f"DEEP mode: Executing {len(plan.steps)} plan steps")
            for i, plan_step in enumerate(plan.steps):
                try:
                    substep_result = await self._execute_substep(plan_step, context, intent)

                    chain.append(ReasoningStep(
                        thought=f"Sub-question {i+1}/{len(plan.steps)}: {substep_result['answer']}",
                        evidence=substep_result['evidence'],
                        confidence=substep_result['confidence'],
                        step_type=StepType.EVIDENCE if i < len(plan.steps) - 1 else StepType.SYNTHESIS
                    ))

                except Exception as e:
                    self.logger.error(f"Substep {i+1} failed: {e}")
                    # Add error step but continue
                    chain.append(ReasoningStep(
                        thought=f"Sub-question {i+1} failed: {str(e)[:100]}",
                        evidence="Error during execution",
                        confidence=0.2,
                        step_type=StepType.EVIDENCE
                    ))

            # Limit total chain length
            if len(chain) > self.max_steps:
                self.logger.warning(
                    f"Chain has {len(chain)} steps, truncating to {self.max_steps}"
                )
                chain = [chain[0]] + chain[-(self.max_steps-1):]

            # Step 3: Multi-pass verification (ERROR BOUNDARY)
            try:
                self.logger.info("DEEP mode: Running multi-pass verification")
                verification = await self.verifier.verify_multipass(chain, context, num_passes=3)

                for pass_num, verify_result in enumerate(verification.passes):
                    if not verify_result.passed:
                        chain.append(ReasoningStep(
                            thought=f"Verification pass {pass_num+1} issue: {verify_result.issue}",
                            evidence=verify_result.correction or "Review recommended",
                            confidence=verify_result.confidence,
                            step_type=StepType.VERIFICATION
                        ))

            except Exception as e:
                self.logger.error(f"Multi-pass verification failed: {e}")
                # Continue without verification

            # Step 4: Backtracking (ERROR BOUNDARY)
            try:
                if hasattr(verification, 'has_contradictions') and verification.has_contradictions:
                    self.logger.info(
                        f"DEEP mode: Resolving {len(verification.contradictions)} contradictions"
                    )

                    revised_chain = await self.backtracker.resolve_contradictions(chain, verification)

                    if len(revised_chain) != len(chain):
                        chain = revised_chain
                        chain.append(ReasoningStep(
                            thought=f"Backtracking complete: resolved contradictions",
                            evidence=f"{len(verification.contradictions)} issues addressed",
                            confidence=0.7,
                            step_type=StepType.BACKTRACK
                        ))

            except Exception as e:
                self.logger.error(f"Backtracking failed: {e}")
                # Continue without backtracking

            # Step 5: Final synthesis
            synthesis = self._synthesize_deep(chain, plan, intent)

            chain.append(ReasoningStep(
                thought=f"Final conclusion: {synthesis['conclusion']}",
                evidence=synthesis['supporting_evidence'],
                confidence=synthesis['confidence'],
                step_type=StepType.SYNTHESIS
            ))

            # Calculate total confidence
            if len(chain) > 3:
                weights = [1.0] * (len(chain) - 3) + [1.5, 1.8, 2.0]
                weighted_conf = sum(s.confidence * w for s, w in zip(chain, weights))
                total_confidence = weighted_conf / sum(weights)
            else:
                total_confidence = sum(s.confidence for s in chain) / len(chain) if chain else 0.0

            return ReasoningResult(
                chain=chain,
                mode=ReasoningMode.DEEP,
                total_confidence=total_confidence,
                metadata={
                    "intent": intent.type.value,
                    "complexity": plan.estimated_complexity,
                    "plan_steps": len(plan.steps)
                }
            )

        except Exception as e:
            self.logger.error(f"DEEP reasoning failed: {e}")

            if self.enable_fallback:
                # Fallback to STANDARD mode
                self.logger.info("Falling back to STANDARD mode")
                return await self.reason_standard(query, features, context)
            else:
                raise

    async def _execute_substep(self, plan_step, context: Any, intent) -> dict:
        """Execute substep with error boundary."""
        try:
            evidence = self.reasoner.extract_evidence(context, intent)

            if not evidence:
                return {
                    'answer': f"Unable to answer: {plan_step.question}",
                    'evidence': "No relevant evidence found",
                    'confidence': 0.3
                }

            synthesis = self.reasoner.synthesize(intent, evidence)

            return {
                'answer': synthesis.reasoning[:200],
                'evidence': "; ".join(synthesis.key_points[:2]),
                'confidence': min(0.9, synthesis.confidence * 0.9)
            }

        except Exception as e:
            self.logger.error(f"Substep execution failed: {e}")
            return {
                'answer': "Substep failed",
                'evidence': str(e)[:100],
                'confidence': 0.2
            }

    def _synthesize_deep(self, chain: List[ReasoningStep], plan, intent) -> dict:
        """Synthesize final conclusion with error handling."""
        try:
            evidence_steps = [s for s in chain if s.step_type == StepType.EVIDENCE]
            synthesis_steps = [s for s in chain if s.step_type == StepType.SYNTHESIS]

            if synthesis_steps:
                latest_synthesis = synthesis_steps[-1].thought
                conclusion = latest_synthesis[:300]
            else:
                conclusion = "Analysis complete based on available evidence"

            supporting_evidence = "; ".join([
                s.evidence[:100] for s in evidence_steps[:3]
                if s.evidence and len(s.evidence) > 10
            ])

            if not supporting_evidence:
                supporting_evidence = "Multiple reasoning steps analyzed"

            if synthesis_steps:
                confidence = sum(s.confidence for s in synthesis_steps) / len(synthesis_steps)
            else:
                confidence = 0.5

            return {
                'conclusion': conclusion,
                'supporting_evidence': supporting_evidence,
                'confidence': min(0.95, confidence)
            }

        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return {
                'conclusion': "Synthesis failed",
                'supporting_evidence': str(e)[:100],
                'confidence': 0.3
            }

    def _create_timeout_fallback(self, query: Any, mode: ReasoningMode) -> ReasoningResult:
        """Create fallback result for timeout."""
        return ReasoningResult(
            chain=[
                ReasoningStep(
                    thought=f"Reasoning timeout in {mode.value} mode",
                    evidence="Operation exceeded time limit, returning fallback",
                    confidence=0.0,
                    step_type=StepType.UNDERSTANDING
                )
            ],
            mode=mode,
            total_confidence=0.0,
            metadata={
                "timeout": True,
                "fallback": True
            }
        )

    def _create_error_fallback(self, query: Any, mode: ReasoningMode, error: str) -> ReasoningResult:
        """Create fallback result for errors."""
        return ReasoningResult(
            chain=[
                ReasoningStep(
                    thought=f"Reasoning failed in {mode.value} mode",
                    evidence=f"Error: {error[:200]}",
                    confidence=0.0,
                    step_type=StepType.UNDERSTANDING
                )
            ],
            mode=mode,
            total_confidence=0.0,
            metadata={
                "error": True,
                "fallback": True,
                "error_message": error[:500]
            }
        )

    def _create_mode_fallback(self, mode: ReasoningMode, error: str) -> ReasoningResult:
        """Create minimal fallback for mode-specific failures."""
        return ReasoningResult(
            chain=[
                ReasoningStep(
                    thought=f"{mode.value} mode failed, using minimal fallback",
                    evidence=f"Error: {error[:100]}",
                    confidence=0.1,
                    step_type=StepType.UNDERSTANDING
                )
            ],
            mode=mode,
            total_confidence=0.1,
            metadata={
                "fallback": True,
                "error": error[:500]
            }
        )

    def estimate_confidence(self, features: Any, context: Any) -> float:
        """Quick confidence estimation."""
        try:
            return self.reasoner.estimate_confidence(features, context)
        except Exception as e:
            self.logger.error(f"Confidence estimation failed: {e}")
            return 0.5

    def select_mode(self, query: Any, features: Any, context: Any) -> ReasoningMode:
        """Select appropriate reasoning mode."""
        try:
            confidence = self.estimate_confidence(features, context)
            intent = self.planner.analyze_intent(query, features)

            if confidence >= 0.85 and intent.complexity < 0.4:
                return ReasoningMode.FAST
            elif confidence < 0.5 or intent.complexity > 0.7:
                return ReasoningMode.DEEP
            else:
                return ReasoningMode.STANDARD

        except Exception as e:
            self.logger.error(f"Mode selection failed: {e}, defaulting to STANDARD")
            return ReasoningMode.STANDARD

    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.

        PHASE 2: Resource monitoring

        Returns:
            Dictionary with resource usage stats
        """
        return self.resource_monitor.get_stats()


# ============================================================================
# Convenience Functions
# ============================================================================

async def reason_with_mode(
    query: Any,
    features: Any,
    context: Any,
    mode: ReasoningMode = ReasoningMode.STANDARD,
    **kwargs
) -> ReasoningResult:
    """Convenience function for one-off reasoning."""
    engine = ReasoningEngine(mode=mode, **kwargs)
    return await engine.reason(query, features, context)


async def auto_reason(
    query: Any,
    features: Any,
    context: Any,
    **kwargs
) -> ReasoningResult:
    """
    Automatically select and apply reasoning mode.

    This is the simplest way to use the reasoning engine.
    """
    engine = ReasoningEngine(**kwargs)
    selected_mode = engine.select_mode(query, features, context)
    return await engine.reason(query, features, context, mode=selected_mode)
