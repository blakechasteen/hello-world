"""
Conscience Protocol
===================
Protocol definition for the Conscience safety system.

Enables drop-in replacement of the default Conscience with custom
implementations (e.g., EnterpriseConscience, MockConscience).

Philosophy: Protocol defines WHAT, not HOW (interface, not implementation).

Author: HoloLoom Team
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

# =============================================================================
# Step Types for Agentic Integration
# =============================================================================

class StepType(IntEnum):
    """
    Types of reasoning steps in agentic pipelines.

    Used to classify the nature of each step for appropriate gating.
    """
    QUERY = 0           # Initial query processing
    VERIFICATION = 1    # Verification loop step
    RESEARCH = 2        # Research sub-query
    PLAN_STEP = 3       # Step in plan-execute mode
    TOOL_EXECUTION = 4  # Tool/action execution
    SYNTHESIS = 5       # Response synthesis


class RiskLevel(IntEnum):
    """
    Risk levels for conscience decisions.

    Compatible with alignment framework's SafetyRiskLevel.
    """
    NONE = 0       # No risk detected
    LOW = 1        # Minor risk, proceed with logging
    MEDIUM = 2     # Moderate risk, recommend review
    HIGH = 3       # High risk, block or escalate
    CRITICAL = 4   # Critical risk, always block


# =============================================================================
# Decision Dataclass
# =============================================================================

@dataclass
class ConscienceDecision:
    """
    Result of a conscience gate check.

    Captures whether an action is allowed, at what risk level,
    and with what guidance for proceeding or escalating.

    Attributes:
        allowed: Whether the action is allowed to proceed
        risk_level: Risk level of the action
        reason: Human-readable explanation
        voice: Voice level (QUIET/WHISPER/VOICE/SHOUT)
        voice_level: Numeric voice level (for comparison)
        concerns: List of concerns raised
        guidance: Suggested course of action
        witness_id: ID of witness record (if witnessed)
        step_type: Type of reasoning step (for agentic)
        step_index: Index of step in sequence (for agentic)
        evaluation_time_ms: Time taken for evaluation
        metadata: Additional context
    """
    allowed: bool
    risk_level: RiskLevel
    reason: str
    voice: str  # Voice name (QUIET, WHISPER, VOICE, SHOUT)
    voice_level: int = 0
    concerns: list[dict[str, Any]] = field(default_factory=list)
    guidance: str = ""
    witness_id: str | None = None
    step_type: StepType = StepType.QUERY
    step_index: int = 0
    evaluation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def should_block(self) -> bool:
        """Whether this decision should block the action."""
        return not self.allowed or self.risk_level >= RiskLevel.HIGH

    @property
    def requires_human(self) -> bool:
        """Whether this decision recommends human review."""
        return self.risk_level >= RiskLevel.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.name,
            "risk_level_value": self.risk_level.value,
            "reason": self.reason,
            "voice": self.voice,
            "voice_level": self.voice_level,
            "concerns": self.concerns,
            "guidance": self.guidance,
            "witness_id": self.witness_id,
            "step_type": self.step_type.name,
            "step_index": self.step_index,
            "evaluation_time_ms": self.evaluation_time_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_safety_gate_decision(self) -> dict[str, Any]:
        """
        Convert to legacy SafetyGateDecision format.

        For backward compatibility with AgenticSafetyAdapter.
        """
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.name,
            "reason": self.reason,
            "guidance": self.guidance,
            "metadata": self.metadata,
        }


# =============================================================================
# Conscience Protocol (Interface)
# =============================================================================

@runtime_checkable
class ConscienceProtocol(Protocol):
    """
    Protocol for Conscience implementations.

    Any class implementing these three methods can be used as a Conscience:
    - consider: Evaluate an action before execution
    - witness: Record an action and its outcome
    - learn: Improve from feedback

    Usage:
        class MyConscience:
            async def consider(self, action, context) -> Judgment:
                ...
            async def witness(self, action, context, judgment, result) -> str:
                ...
            async def learn(self, feedback) -> None:
                ...

        # Type check
        assert isinstance(MyConscience(), ConscienceProtocol)
    """

    async def consider(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> Any:  # Returns Judgment
        """
        Consider an action before execution.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            Judgment with voice level and concerns
        """
        ...

    async def witness(
        self,
        action: str,
        context: dict[str, Any],
        judgment: Any,  # Judgment type
        result: dict[str, Any] | None = None
    ) -> str:
        """
        Witness an action and its outcome.

        Args:
            action: The action that was taken
            context: Context for the action
            judgment: The prior judgment
            result: The outcome of the action

        Returns:
            Record ID for this observation
        """
        ...

    async def learn(
        self,
        feedback: dict[str, Any]
    ) -> None:
        """
        Learn from feedback.

        Args:
            feedback: Feedback on past judgments/actions
        """
        ...


# =============================================================================
# Optional Extended Protocol
# =============================================================================

@runtime_checkable
class ExtendedConscienceProtocol(ConscienceProtocol, Protocol):
    """
    Extended protocol with additional convenience methods.

    Optional - not required for basic integration.
    """

    async def is_safe(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Quick check if an action is safe."""
        ...

    async def gate(
        self,
        action: str,
        context: dict[str, Any] | None = None,
        execute_fn=None,
    ) -> dict[str, Any]:
        """Gate an action through conscience with automatic execution."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about conscience activity."""
        ...


# =============================================================================
# Factory Functions
# =============================================================================

def create_allowed_decision(
    reason: str = "No concerns raised",
    step_type: StepType = StepType.QUERY,
    step_index: int = 0,
    evaluation_time_ms: float = 0.0,
) -> ConscienceDecision:
    """
    Create an allowed ConscienceDecision (safe to proceed).

    Args:
        reason: Explanation for allowing
        step_type: Type of reasoning step
        step_index: Index of step in sequence
        evaluation_time_ms: Time taken for evaluation

    Returns:
        ConscienceDecision with allowed=True
    """
    return ConscienceDecision(
        allowed=True,
        risk_level=RiskLevel.NONE,
        reason=reason,
        voice="QUIET",
        voice_level=0,
        guidance="Proceed normally",
        step_type=step_type,
        step_index=step_index,
        evaluation_time_ms=evaluation_time_ms,
    )


def create_blocked_decision(
    reason: str,
    risk_level: RiskLevel = RiskLevel.HIGH,
    concerns: list[dict[str, Any]] | None = None,
    guidance: str = "Action blocked for safety",
    step_type: StepType = StepType.QUERY,
    step_index: int = 0,
    evaluation_time_ms: float = 0.0,
) -> ConscienceDecision:
    """
    Create a blocked ConscienceDecision.

    Args:
        reason: Explanation for blocking
        risk_level: Risk level that caused blocking
        concerns: List of concerns
        guidance: Guidance for handling the block
        step_type: Type of reasoning step
        step_index: Index of step in sequence
        evaluation_time_ms: Time taken for evaluation

    Returns:
        ConscienceDecision with allowed=False
    """
    voice = "SHOUT" if risk_level >= RiskLevel.HIGH else "VOICE"
    voice_level = 3 if risk_level >= RiskLevel.HIGH else 2

    return ConscienceDecision(
        allowed=False,
        risk_level=risk_level,
        reason=reason,
        voice=voice,
        voice_level=voice_level,
        concerns=concerns or [],
        guidance=guidance,
        step_type=step_type,
        step_index=step_index,
        evaluation_time_ms=evaluation_time_ms,
    )


def create_review_decision(
    reason: str,
    concerns: list[dict[str, Any]] | None = None,
    guidance: str = "Human review recommended",
    step_type: StepType = StepType.QUERY,
    step_index: int = 0,
    evaluation_time_ms: float = 0.0,
) -> ConscienceDecision:
    """
    Create a ConscienceDecision requiring human review.

    Args:
        reason: Explanation for review requirement
        concerns: List of concerns
        guidance: Guidance for review
        step_type: Type of reasoning step
        step_index: Index of step in sequence
        evaluation_time_ms: Time taken for evaluation

    Returns:
        ConscienceDecision with allowed=True but requires_human=True
    """
    return ConscienceDecision(
        allowed=True,
        risk_level=RiskLevel.MEDIUM,
        reason=reason,
        voice="VOICE",
        voice_level=2,
        concerns=concerns or [],
        guidance=guidance,
        step_type=step_type,
        step_index=step_index,
        evaluation_time_ms=evaluation_time_ms,
    )


# =============================================================================
# Null Implementation (No-op Conscience)
# =============================================================================

class NullConscience:
    """
    No-op Conscience implementation.

    Always allows actions, never raises concerns.
    Useful for testing or when conscience is disabled.
    """

    async def consider(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Always returns safe judgment."""
        return {
            "voice": "QUIET",
            "voice_level": 0,
            "concerns": [],
            "summary": "Conscience disabled",
            "guidance": "Proceed normally",
            "allowed": True,
        }

    async def witness(
        self,
        action: str,
        context: dict[str, Any],
        judgment: Any,
        result: dict[str, Any] | None = None
    ) -> str:
        """No-op witness, returns empty record ID."""
        return ""

    async def learn(
        self,
        feedback: dict[str, Any]
    ) -> None:
        """No-op learn."""
        pass

    async def is_safe(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Always safe."""
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Empty statistics."""
        return {"status": "disabled"}
