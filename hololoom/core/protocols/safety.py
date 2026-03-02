"""
Safety Gate Protocol for Agentic Reasoning
==========================================
Defines the interface for safety gating in agentic systems.

Philosophy:
- Protocol defines WHAT (interface), not HOW (implementation)
- Single-method interface for maximum clarity and auditability
- All implementations are swappable via dependency injection
- Enables testing with mocks

Usage:
    from hololoom.protocols.safety import SafetyGateProtocol

    class MyCustomSafetyGate:
        async def gate_reasoning(
            self, query_text, reasoning_mode, epistemic_confidence
        ) -> SafetyDecision:
            ...

    # Use with AgenticOrchestrator
    orchestrator = AgenticOrchestrator(
        learning_engine=engine,
        safety_gate=MyCustomSafetyGate()
    )

Author: MRF Swarm (VERIFY + CRITIQUE + ELEGANCE perspectives)
Date: 2025-12-01 (Safe Integration Phase)
"""

from typing import Protocol, runtime_checkable, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Safety Types (Re-exported for convenience)
# ============================================================================

class SafetyRiskLevel(Enum):
    """Risk level classification for safety decisions."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyGateDecision:
    """
    Result of safety gate evaluation.

    Designed for agentic reasoning context with epistemic awareness.
    """
    allowed: bool
    risk_level: SafetyRiskLevel
    reason: str
    requires_approval: bool = False

    # Epistemic context (Consciousness Integration)
    epistemic_confidence: Optional[float] = None
    epistemic_warning: Optional[str] = None

    # Metadata for audit trail
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def blocked(self) -> bool:
        """Alias for readability."""
        return not self.allowed


# ============================================================================
# Safety Gate Protocol
# ============================================================================

@runtime_checkable
class SafetyGateProtocol(Protocol):
    """
    Protocol for safety gating in agentic reasoning.

    Single-method interface for maximum clarity:
    - Input: What the agent wants to do (query, mode, context)
    - Output: Whether it's safe to proceed (decision)

    The protocol is intentionally simple - all complexity lives in implementations.

    Example implementation:
        class MyGate:
            async def gate_reasoning(
                self, query_text, reasoning_mode, epistemic_confidence, context
            ) -> SafetyGateDecision:
                # Check adversarial patterns
                # Evaluate risk based on mode
                # Consider epistemic confidence
                return SafetyGateDecision(allowed=True, ...)
    """

    async def gate_reasoning(
        self,
        query_text: str,
        reasoning_mode: str,
        epistemic_confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyGateDecision:
        """
        Gate agentic reasoning based on safety evaluation.

        Args:
            query_text: The query text to evaluate
            reasoning_mode: Reasoning mode (direct, verify, research, plan_execute)
            epistemic_confidence: Optional epistemic confidence from awareness layer (0.0-1.0)
                - Low values indicate the system is uncertain about its knowledge
                - Should escalate risk assessment when low
            context: Optional additional context for evaluation

        Returns:
            SafetyGateDecision indicating whether reasoning should proceed
        """
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SafetyGateProtocol',
    'SafetyGateDecision',
    'SafetyRiskLevel',
]
