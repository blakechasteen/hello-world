from __future__ import annotations
"""
Agentic Conscience Adapter
==========================
Bridges the agentic reasoning system with the Conscience architecture.

Unlike the safety_adapter which gates only at reason() entry, this adapter
provides per-step gating for all 4 reasoning modes (DIRECT, VERIFY, RESEARCH,
PLAN_EXECUTE) to close the gap where sub-queries bypass safety checks.

Key Features:
- Per-step conscience gating via gate_step()
- Witnesses outcomes via witness_step()
- Learns from feedback via learn_from_feedback()
- Graceful degradation when conscience unavailable
- Full backward compatibility with AgenticSafetyAdapter

Design Philosophy:
- "A conscience that speaks, not shouts. That guides, not gates."
- Fail-open for availability (with logging)
- Voice levels guide response (QUIET → SHOUT)

Author: HoloLoom Team
Date: 2025-12-03
"""

import logging
import time
from typing import Any, Optional

from hololoom.protocols.conscience import (
    ConscienceDecision,
    ConscienceProtocol,
    NullConscience,
    RiskLevel,
    StepType,
    create_allowed_decision,
    create_blocked_decision,
)

# Import ConscienceCalibrator for Thompson Sampling calibration
try:
    from hololoom.agentic.conscience_calibrator import (
        CalibrationEvent,
        ConscienceCalibrator,
        create_calibrator,
    )
    CALIBRATOR_AVAILABLE = True
except ImportError:
    CALIBRATOR_AVAILABLE = False
    ConscienceCalibrator = None
    create_calibrator = None
    CalibrationEvent = None

# Import Conscience with graceful degradation
try:
    from hololoom.conscience.core import Conscience
    from hololoom.conscience.judgment import Concern, Judgment, Voice
    from hololoom.conscience.lenses import (
        CompositeLens,
        DeceptionLens,
        HarmLens,
        PowerLens,
        paranoid,
        research,
        standard,
    )
    CONSCIENCE_AVAILABLE = True
except ImportError:
    CONSCIENCE_AVAILABLE = False
    Conscience = None
    Voice = None
    Judgment = None
    Concern = None
    CompositeLens = None
    HarmLens = None
    DeceptionLens = None
    PowerLens = None

    def standard():
        return None

    def paranoid():
        return None

    def research():
        return None


logger = logging.getLogger("hololoom.agentic.conscience_adapter")


# =============================================================================
# Mode-to-StepType Mapping
# =============================================================================

# Map agentic reasoning modes to default step types
MODE_TO_STEP_TYPE: dict[str, StepType] = {
    "direct": StepType.QUERY,
    "verify": StepType.VERIFICATION,
    "research": StepType.RESEARCH,
    "plan_execute": StepType.PLAN_STEP,
}


# =============================================================================
# Main Adapter
# =============================================================================

class AgenticConscienceAdapter:
    """
    Adapter that integrates Conscience with agentic reasoning.

    Provides per-step safety gating across all 4 reasoning modes:
    - DIRECT: Single query gating
    - VERIFY: Per-verification-loop gating
    - RESEARCH: Per-sub-query gating
    - PLAN_EXECUTE: Per-plan-step gating

    Key Methods:
    - gate_reasoning(): Gate at reason() entry (like safety_adapter)
    - gate_step(): Gate individual steps within multi-step reasoning
    - witness_step(): Record step outcomes for audit trail
    - learn_from_feedback(): Learn from explicit feedback

    Example:
        conscience = Conscience(lens=standard())
        adapter = AgenticConscienceAdapter(conscience)

        # Gate at entry
        decision = await adapter.gate_reasoning(
            query="What is Thompson Sampling?",
            mode="verify"
        )

        if decision.allowed:
            # Gate each step
            for step in verification_steps:
                step_decision = await adapter.gate_step(
                    step_action=step.query,
                    step_type=StepType.VERIFICATION,
                    step_index=i,
                    parent_decision=decision
                )

                if step_decision.allowed:
                    result = await execute_step(step)
                    await adapter.witness_step(step_decision, result, success=True)
    """

    def __init__(
        self,
        conscience: ConscienceProtocol | None = None,
        preset: str = "standard",
        auto_create: bool = True,
        fail_open: bool = True,
        calibrator: Optional['ConscienceCalibrator'] = None,
        enable_calibration: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            conscience: Conscience instance to use. If None and auto_create=True,
                       creates conscience with specified preset.
            preset: Lens preset if auto-creating ("standard", "paranoid", "research")
            auto_create: If True and conscience is None, auto-create with preset.
            fail_open: If True, allow when conscience unavailable (with logging).
                      If False, block when conscience unavailable.
            calibrator: Optional ConscienceCalibrator for Thompson Sampling calibration.
                       If None and enable_calibration=True, auto-creates calibrator.
            enable_calibration: If True, enables Thompson Sampling calibration.
        """
        self._fail_open = fail_open
        self._available = False
        self._conscience: ConscienceProtocol | None = None
        self._calibrator: ConscienceCalibrator | None = None
        self._enable_calibration = enable_calibration

        if not CONSCIENCE_AVAILABLE:
            logger.warning(
                "Conscience framework not available. "
                "AgenticConscienceAdapter will operate in pass-through mode."
            )
            if not fail_open:
                logger.warning("fail_open=False but Conscience unavailable")
            return

        if conscience is not None:
            self._conscience = conscience
            self._available = True
            logger.info("Conscience adapter initialized with provided conscience")
        elif auto_create:
            # Create with specified preset
            lens = self._get_preset_lens(preset)
            if lens is not None:
                self._conscience = Conscience(lens=lens)
                self._available = True
                logger.info(f"Conscience adapter auto-created with '{preset}' preset")
            else:
                logger.warning(f"Failed to create lens for preset '{preset}'")

        # Initialize calibrator for Thompson Sampling calibration
        if enable_calibration and CALIBRATOR_AVAILABLE:
            if calibrator is not None:
                self._calibrator = calibrator
                logger.info("Conscience adapter initialized with provided calibrator")
            else:
                # Auto-create calibrator with sensible defaults
                self._calibrator = create_calibrator(
                    min_updates_for_calibration=10,
                    prior_alpha=2.0,  # Slight bias toward allowing
                    prior_beta=1.0,
                    drift_threshold=0.1,
                )
                logger.info("Conscience adapter auto-created calibrator")
        elif enable_calibration and not CALIBRATOR_AVAILABLE:
            logger.warning(
                "Calibration enabled but ConscienceCalibrator not available. "
                "Thompson Sampling calibration disabled."
            )

    @property
    def available(self) -> bool:
        """Check if conscience adapter is fully available."""
        return self._available

    @property
    def conscience(self) -> ConscienceProtocol | None:
        """Get the underlying conscience (for advanced use)."""
        return self._conscience

    @property
    def calibrator(self) -> Optional['ConscienceCalibrator']:
        """Get the calibrator for Thompson Sampling calibration."""
        return self._calibrator

    @property
    def calibration_enabled(self) -> bool:
        """Check if calibration is enabled and available."""
        return self._enable_calibration and self._calibrator is not None

    def _get_preset_lens(self, preset: str) -> Optional['CompositeLens']:
        """Get lens for a preset name."""
        if not CONSCIENCE_AVAILABLE:
            return None

        presets = {
            "standard": standard,
            "paranoid": paranoid,
            "research": research,
        }

        factory = presets.get(preset.lower())
        if factory:
            return factory()

        logger.warning(f"Unknown preset '{preset}', using 'standard'")
        return standard()

    def _voice_to_risk_level(self, voice: 'Voice') -> RiskLevel:
        """Map Voice level to RiskLevel."""
        if not CONSCIENCE_AVAILABLE:
            return RiskLevel.NONE

        mapping = {
            Voice.QUIET: RiskLevel.NONE,
            Voice.WHISPER: RiskLevel.LOW,
            Voice.VOICE: RiskLevel.MEDIUM,
            Voice.SHOUT: RiskLevel.HIGH,
        }
        return mapping.get(voice, RiskLevel.MEDIUM)

    def _judgment_to_decision(
        self,
        judgment: 'Judgment',
        step_type: StepType = StepType.QUERY,
        step_index: int = 0,
        evaluation_time_ms: float = 0.0,
        witness_id: str | None = None,
    ) -> ConscienceDecision:
        """
        Convert Judgment to ConscienceDecision.

        Handles both Judgment dataclass objects and dict representations
        (e.g., from NullConscience or dict-based implementations).
        """
        if not CONSCIENCE_AVAILABLE:
            return create_allowed_decision(
                reason="Conscience unavailable",
                step_type=step_type,
                step_index=step_index,
            )

        # Handle dict representation (e.g., from NullConscience)
        if isinstance(judgment, dict):
            # Extract values from dict with sensible defaults
            voice_value = judgment.get('voice', 'QUIET')
            voice_level = judgment.get('voice_level', 0)
            concerns = judgment.get('concerns', [])
            allowed = judgment.get('allowed', True)
            summary = judgment.get('summary', judgment.get('reason', 'No concerns'))
            guidance = judgment.get('guidance', 'Proceed normally')

            # Map voice string to RiskLevel
            voice_to_risk = {
                'QUIET': RiskLevel.NONE,
                'WHISPER': RiskLevel.LOW,
                'VOICE': RiskLevel.MEDIUM,
                'SHOUT': RiskLevel.HIGH,
            }
            risk_level = voice_to_risk.get(
                voice_value if isinstance(voice_value, str) else 'QUIET',
                RiskLevel.NONE
            )

            return ConscienceDecision(
                allowed=allowed,
                risk_level=risk_level,
                reason=summary,
                voice=voice_value if isinstance(voice_value, str) else 'QUIET',
                voice_level=voice_level if isinstance(voice_level, int) else 0,
                concerns=concerns if isinstance(concerns, list) else [],
                guidance=guidance,
                witness_id=witness_id,
                step_type=step_type,
                step_index=step_index,
                evaluation_time_ms=evaluation_time_ms,
                metadata={
                    "source": "dict_judgment",
                    "concern_count": len(concerns) if isinstance(concerns, list) else 0,
                },
            )

        # Handle Judgment dataclass object
        # Convert concerns to dict format
        concerns_list = []
        for concern in judgment.concerns:
            concerns_list.append({
                "lens": concern.lens,
                "category": concern.category,
                "description": concern.description,
                "confidence": concern.confidence,
                "suggested_mitigation": concern.suggested_mitigation,
            })

        return ConscienceDecision(
            allowed=judgment.allowed,
            risk_level=self._voice_to_risk_level(judgment.voice),
            reason=judgment.summary,
            voice=judgment.voice.name,
            voice_level=judgment.voice.value,
            concerns=concerns_list,
            guidance=judgment.guidance,
            witness_id=witness_id,
            step_type=step_type,
            step_index=step_index,
            evaluation_time_ms=evaluation_time_ms,
            metadata={
                "judgment_timestamp": judgment.timestamp.isoformat(),
                "concern_count": len(judgment.concerns),
                "top_concern": (
                    judgment.top_concern.description
                    if judgment.top_concern else None
                ),
            },
        )

    # =========================================================================
    # Core API
    # =========================================================================

    async def gate_reasoning(
        self,
        query: str,
        mode: str,
        context: dict[str, Any] | None = None,
        epistemic_confidence: float | None = None,
    ) -> ConscienceDecision:
        """
        Gate agentic reasoning at reason() entry.

        This is the top-level gate before any reasoning begins.
        Similar to AgenticSafetyAdapter.gate_reasoning() but returns
        ConscienceDecision with richer information.

        Args:
            query: The query text to evaluate
            mode: Reasoning mode (direct, verify, research, plan_execute)
            context: Optional additional context
            epistemic_confidence: Optional epistemic confidence (0.0-1.0)

        Returns:
            ConscienceDecision indicating whether reasoning should proceed
        """
        start_time = time.perf_counter()

        # Graceful degradation
        if not self._available:
            if self._fail_open:
                logger.debug("Conscience unavailable, allowing by default (fail_open=True)")
                return create_allowed_decision(
                    reason="Conscience unavailable - allowing with caution",
                    step_type=MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY),
                )
            else:
                logger.warning("Conscience unavailable and fail_open=False, blocking")
                return create_blocked_decision(
                    reason="Conscience unavailable and fail_open=False",
                    risk_level=RiskLevel.HIGH,
                    step_type=MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY),
                )

        try:
            # Build context
            full_context = {
                "mode": mode,
                "step_type": "reason_entry",
                **(context or {}),
            }

            if epistemic_confidence is not None:
                full_context["epistemic_confidence"] = epistemic_confidence

            # Consult conscience
            judgment = await self._conscience.consider(query, full_context)

            # Calculate evaluation time
            evaluation_time_ms = (time.perf_counter() - start_time) * 1000

            decision = self._judgment_to_decision(
                judgment,
                step_type=MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY),
                step_index=0,
                evaluation_time_ms=evaluation_time_ms,
            )

            # Apply Thompson Sampling calibration if enabled
            if self.calibration_enabled:
                step_type = MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY)
                decision = self._calibrator.calibrate_decision(decision, step_type)
                decision.metadata["calibration_applied"] = True
                decision.metadata["calibration_confidence"] = (
                    self._calibrator.get_confidence(step_type)
                )

            return decision

        except Exception as e:
            logger.error(f"Conscience evaluation failed: {e}")
            if self._fail_open:
                return create_allowed_decision(
                    reason=f"Conscience error - allowing with caution: {type(e).__name__}",
                    step_type=MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY),
                )
            else:
                return create_blocked_decision(
                    reason=f"Conscience error: {type(e).__name__}",
                    risk_level=RiskLevel.CRITICAL,
                    step_type=MODE_TO_STEP_TYPE.get(mode.lower(), StepType.QUERY),
                )

    async def gate_step(
        self,
        step_action: str,
        step_type: StepType,
        step_index: int,
        parent_decision: ConscienceDecision | None = None,
        context: dict[str, Any] | None = None,
    ) -> ConscienceDecision:
        """
        Gate an individual step within multi-step reasoning.

        This is the per-step gate that closes the gap where sub-queries
        (in VERIFY, RESEARCH, PLAN_EXECUTE modes) bypass safety checks.

        Args:
            step_action: The action/query for this step
            step_type: Type of reasoning step
            step_index: Index of this step in the sequence
            parent_decision: Decision from gate_reasoning() (for context)
            context: Optional additional context

        Returns:
            ConscienceDecision for this specific step
        """
        start_time = time.perf_counter()

        # Graceful degradation
        if not self._available:
            if self._fail_open:
                return create_allowed_decision(
                    reason="Conscience unavailable",
                    step_type=step_type,
                    step_index=step_index,
                )
            else:
                return create_blocked_decision(
                    reason="Conscience unavailable",
                    risk_level=RiskLevel.HIGH,
                    step_type=step_type,
                    step_index=step_index,
                )

        try:
            # Build context with parent decision info
            full_context = {
                "step_type": step_type.name,
                "step_index": step_index,
                **(context or {}),
            }

            if parent_decision:
                full_context["parent_voice"] = parent_decision.voice
                full_context["parent_risk_level"] = parent_decision.risk_level.name

            # Consult conscience
            judgment = await self._conscience.consider(step_action, full_context)

            # Calculate evaluation time
            evaluation_time_ms = (time.perf_counter() - start_time) * 1000

            decision = self._judgment_to_decision(
                judgment,
                step_type=step_type,
                step_index=step_index,
                evaluation_time_ms=evaluation_time_ms,
            )

            # Apply Thompson Sampling calibration if enabled
            if self.calibration_enabled:
                decision = self._calibrator.calibrate_decision(decision, step_type)
                decision.metadata["calibration_applied"] = True
                decision.metadata["calibration_confidence"] = (
                    self._calibrator.get_confidence(step_type)
                )

            return decision

        except Exception as e:
            logger.error(f"Conscience step evaluation failed: {e}")
            if self._fail_open:
                return create_allowed_decision(
                    reason=f"Conscience error: {type(e).__name__}",
                    step_type=step_type,
                    step_index=step_index,
                )
            else:
                return create_blocked_decision(
                    reason=f"Conscience error: {type(e).__name__}",
                    risk_level=RiskLevel.HIGH,
                    step_type=step_type,
                    step_index=step_index,
                )

    async def witness_step(
        self,
        decision: ConscienceDecision,
        result: dict[str, Any],
        success: bool,
        weight: float = 1.0,
    ) -> str:
        """
        Witness a step's outcome for audit trail and learning.

        This method serves two purposes:
        1. Records the outcome for audit trail (conscience witness)
        2. Updates Thompson Sampling calibration (if enabled)

        Args:
            decision: The ConscienceDecision that led to this step
            result: The outcome of the step
            success: Whether the step was successful
            weight: Weight for calibration update (default 1.0)

        Returns:
            Witness record ID (empty if conscience unavailable)
        """
        # Update calibrator with outcome (works even if conscience unavailable)
        calibration_event = None
        if self.calibration_enabled:
            calibration_event = self._calibrator.update(
                decision=decision,
                success=success,
                weight=weight,
            )
            if calibration_event:
                logger.debug(
                    f"Calibration updated: step_type={decision.step_type.name}, "
                    f"success={success}, new_mean={calibration_event.prior_after.mean:.3f}"
                )

        if not self._available:
            return ""

        try:
            # Build judgment-like object for witness
            # (Conscience.witness expects a Judgment, but we have ConscienceDecision)
            witness_result = {
                "success": success,
                "confidence": result.get("confidence", 0.8 if success else 0.3),
                "step_type": decision.step_type.name,
                "step_index": decision.step_index,
                **result,
            }

            # Add calibration metadata if available
            if calibration_event:
                witness_result["calibration"] = {
                    "prior_mean_before": calibration_event.prior_before.mean,
                    "prior_mean_after": calibration_event.prior_after.mean,
                    "confidence": calibration_event.prior_after.confidence,
                }

            # Witness through conscience
            if CONSCIENCE_AVAILABLE and hasattr(self._conscience, 'witness'):
                # Create minimal judgment for witness
                from hololoom.conscience.judgment import Voice

                # Reconstruct judgment from decision
                voice = getattr(Voice, decision.voice, Voice.QUIET)
                judgment = Judgment(
                    voice=voice,
                    concerns=[],  # Concerns already recorded in decision
                    summary=decision.reason,
                    guidance=decision.guidance,
                )

                record_id = await self._conscience.witness(
                    action=f"step_{decision.step_type.name}_{decision.step_index}",
                    context={
                        "step_type": decision.step_type.name,
                        "step_index": decision.step_index,
                    },
                    judgment=judgment,
                    result=witness_result,
                )
                return record_id

            return ""

        except Exception as e:
            logger.warning(f"Failed to witness step: {e}")
            return ""

    async def learn_from_feedback(
        self,
        feedback: dict[str, Any],
        decisions: list[ConscienceDecision] | None = None,
    ) -> None:
        """
        Learn from explicit feedback on past decisions.

        This method serves two purposes:
        1. Teaches the conscience from feedback
        2. Updates Thompson Sampling calibration based on decisions

        Args:
            feedback: Feedback dictionary containing:
                - correct: Was the overall outcome correct?
                - confidence: Confidence in feedback (0.0-1.0)
                - record_ids: Optional list of specific witness records
                - pattern: Optional pattern to learn (e.g., "harm:code_execution")
            decisions: Optional list of decisions to learn from
        """
        # Update calibrator from decisions (works even if conscience unavailable)
        if self.calibration_enabled and decisions:
            correct = feedback.get("correct", True)
            weight = feedback.get("confidence", 1.0)

            for decision in decisions:
                self._calibrator.update(
                    decision=decision,
                    success=correct,
                    weight=weight,
                )

            logger.debug(
                f"Calibration updated from {len(decisions)} decisions, "
                f"correct={correct}, weight={weight}"
            )

        if not self._available:
            return

        try:
            if CONSCIENCE_AVAILABLE and hasattr(self._conscience, 'learn'):
                # Augment feedback with decision metadata
                if decisions:
                    feedback["decision_count"] = len(decisions)
                    feedback["step_types"] = list(set(
                        d.step_type.name for d in decisions
                    ))
                    feedback["voice_levels"] = list(set(
                        d.voice for d in decisions
                    ))

                    # Add calibration statistics
                    if self.calibration_enabled:
                        feedback["calibration_stats"] = (
                            self._calibrator.get_statistics()
                        )

                await self._conscience.learn(feedback)

        except Exception as e:
            logger.warning(f"Failed to learn from feedback: {e}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def is_safe(
        self,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Quick check if an action is safe (QUIET or WHISPER).

        Args:
            action: Action to check
            context: Optional context

        Returns:
            True if action is safe (voice <= WHISPER)
        """
        if not self._available:
            return self._fail_open

        try:
            judgment = await self._conscience.consider(action, context or {})
            return judgment.voice <= Voice.WHISPER

        except Exception:
            return self._fail_open

    def get_statistics(self) -> dict[str, Any]:
        """Get conscience and calibration statistics."""
        stats = {
            "status": "available" if self._available else "unavailable",
            "fail_open": self._fail_open,
            "calibration_enabled": self.calibration_enabled,
        }

        # Add calibration statistics
        if self.calibration_enabled:
            stats["calibration"] = self._calibrator.get_statistics()
            stats["drift_alerts"] = [
                {
                    "step_type": alert.step_type.name,
                    "drift": alert.drift,
                    "direction": alert.direction,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self._calibrator.get_drift_alerts()
            ]

        if not self._available:
            return stats

        try:
            if hasattr(self._conscience, 'get_statistics'):
                stats["conscience"] = self._conscience.get_statistics()
            elif hasattr(self._conscience, '_witness'):
                witness = self._conscience._witness
                if hasattr(witness, 'get_statistics'):
                    stats["conscience"] = witness.get_statistics()

        except Exception as e:
            stats["conscience_error"] = str(e)

        return stats

    def get_calibration_statistics(self) -> dict[str, Any]:
        """Get calibration-specific statistics."""
        if not self.calibration_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            **self._calibrator.get_statistics(),
            "drift_alerts": [
                {
                    "step_type": alert.step_type.name,
                    "drift": alert.drift,
                    "direction": alert.direction,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self._calibrator.get_drift_alerts()
            ],
        }

    def get_drift_alerts(self) -> list[dict[str, Any]]:
        """Get any calibration drift alerts."""
        if not self.calibration_enabled:
            return []

        return [
            {
                "step_type": alert.step_type.name,
                "drift": alert.drift,
                "direction": alert.direction,
                "timestamp": alert.timestamp.isoformat(),
                "mean_before": alert.mean_before,
                "mean_after": alert.mean_after,
            }
            for alert in self._calibrator.get_drift_alerts()
        ]

    def clear_drift_alerts(self) -> None:
        """Clear all drift alerts."""
        if self.calibration_enabled:
            self._calibrator.clear_drift_alerts()

    async def save_calibration(self, path: str) -> None:
        """Save calibration state to disk."""
        if self.calibration_enabled:
            self._calibrator.save(path)
            logger.info(f"Calibration state saved to {path}")

    async def load_calibration(self, path: str) -> bool:
        """Load calibration state from disk."""
        if not self.calibration_enabled:
            logger.warning("Calibration not enabled, cannot load state")
            return False

        if self._calibrator.load(path):
            logger.info(f"Calibration state loaded from {path}")
            return True
        else:
            logger.warning(f"Failed to load calibration state from {path}")
            return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_conscience_adapter(
    conscience: ConscienceProtocol | None = None,
    preset: str = "standard",
    auto_create: bool = True,
    fail_open: bool = True,
    calibrator: Optional['ConscienceCalibrator'] = None,
    enable_calibration: bool = True,
) -> AgenticConscienceAdapter:
    """
    Create an AgenticConscienceAdapter with sensible defaults.

    Args:
        conscience: Optional Conscience to use
        preset: Lens preset ("standard", "paranoid", "research")
        auto_create: If True, auto-create conscience if None provided
        fail_open: If True, allow when conscience unavailable
        calibrator: Optional ConscienceCalibrator for Thompson Sampling
        enable_calibration: If True, enables Thompson Sampling calibration

    Returns:
        Configured AgenticConscienceAdapter
    """
    return AgenticConscienceAdapter(
        conscience=conscience,
        preset=preset,
        auto_create=auto_create,
        fail_open=fail_open,
        calibrator=calibrator,
        enable_calibration=enable_calibration,
    )


def create_null_adapter() -> AgenticConscienceAdapter:
    """
    Create an adapter with NullConscience (no-op, always allows).

    Useful for testing or when conscience should be disabled.
    Calibration is disabled since this is a no-op adapter.

    Returns:
        AgenticConscienceAdapter with NullConscience
    """
    return AgenticConscienceAdapter(
        conscience=NullConscience(),
        auto_create=False,
        fail_open=True,
        enable_calibration=False,  # Disable for null adapter
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main adapter
    "AgenticConscienceAdapter",

    # Factory functions
    "create_conscience_adapter",
    "create_null_adapter",

    # Types (re-export for convenience)
    "ConscienceDecision",
    "StepType",
    "RiskLevel",

    # Availability flags
    "CONSCIENCE_AVAILABLE",
    "CALIBRATOR_AVAILABLE",

    # Mode mapping
    "MODE_TO_STEP_TYPE",
]
