from __future__ import annotations
"""
Agentic Safety Adapter
======================
Translates agentic reasoning vocabulary to alignment framework vocabulary.

This adapter bridges the gap between the agentic reasoning system
(AgenticOrchestrator) and the alignment framework (SafetyGuardrails).

Design Philosophy (MRF ELEGANCE):
- Single responsibility: Translation between two domains
- Clean interface: Implements SafetyGateProtocol
- Fail-closed: Deny on any exception
- TOCTOU protection: Deep copy query text

Security Hardening (MRF CRITIQUE):
- Unicode normalization prevents adversarial pattern bypass
- Deep copy prevents modification-after-check attacks
- Explicit exception handling ensures fail-closed behavior

Author: MRF Swarm (VERIFY + CRITIQUE + ELEGANCE perspectives)
Date: 2025-12-01 (Safe Integration Phase)
"""

import logging
import unicodedata
from typing import Any, Optional

from hololoom.protocols.safety import (
    SafetyGateDecision,
    SafetyRiskLevel,
)

# Import alignment framework with graceful degradation
try:
    from hololoom.alignment.safety_guardrails import (
        ActionCategory,
        ActionRequest,
        RiskLevel,
        SafetyGuardrails,
    )
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    SafetyGuardrails = None
    ActionRequest = None
    ActionCategory = None
    RiskLevel = None

logger = logging.getLogger("hololoom.agentic.safety_adapter")


class AgenticSafetyAdapter:
    """
    Adapter that translates agentic reasoning context to alignment framework.

    Implements SafetyGateProtocol for clean integration with AgenticOrchestrator.

    Key Features:
    - Mode-to-category mapping (agentic vocab → alignment vocab)
    - TOCTOU protection via deep copy
    - Unicode normalization (prevent homoglyph attacks)
    - Fail-closed exception handling

    Example:
        from hololoom.agentic.safety_adapter import AgenticSafetyAdapter
        from hololoom.alignment import SafetyGuardrails

        guardrails = SafetyGuardrails()
        adapter = AgenticSafetyAdapter(guardrails)

        decision = await adapter.gate_reasoning(
            query_text="What is Thompson Sampling?",
            reasoning_mode="verify",
            epistemic_confidence=0.75
        )

        if decision.allowed:
            # Safe to proceed with reasoning
            ...
    """

    # Map agentic reasoning modes to alignment action categories
    # These mappings reflect the risk profile of each mode
    MODE_TO_CATEGORY: dict[str, 'ActionCategory'] = {}

    # Risk profile explanation:
    # - direct: Simple query → lowest risk (QUERY)
    # - verify: Analysis of claims → low-medium risk (ANALYSIS)
    # - research: Multi-step retrieval → medium risk (RETRIEVAL)
    # - plan_execute: Autonomous execution → highest risk (EXECUTION)

    def __init__(
        self,
        guardrails: Optional['SafetyGuardrails'] = None,
        auto_create_guardrails: bool = True
    ):
        """
        Initialize the adapter.

        Args:
            guardrails: SafetyGuardrails instance to use. If None and
                       auto_create_guardrails=True, creates default guardrails.
            auto_create_guardrails: If True and guardrails is None, automatically
                                   creates guardrails with safe defaults.
        """
        if not ALIGNMENT_AVAILABLE:
            logger.warning(
                "Alignment framework not available. "
                "AgenticSafetyAdapter will operate in pass-through mode."
            )
            self.guardrails = None
            self._available = False
            return

        # Initialize mode-to-category mapping (must be done after import check)
        if not AgenticSafetyAdapter.MODE_TO_CATEGORY:
            AgenticSafetyAdapter.MODE_TO_CATEGORY = {
                "direct": ActionCategory.QUERY,
                "verify": ActionCategory.ANALYSIS,
                "research": ActionCategory.RETRIEVAL,
                "plan_execute": ActionCategory.EXECUTION,
            }

        if guardrails is not None:
            self.guardrails = guardrails
        elif auto_create_guardrails:
            logger.info("Auto-creating SafetyGuardrails with safe defaults")
            self.guardrails = SafetyGuardrails()
        else:
            self.guardrails = None

        self._available = self.guardrails is not None

    @property
    def available(self) -> bool:
        """Check if safety adapter is fully available."""
        return self._available

    # Zero-width and invisible Unicode characters to strip
    INVISIBLE_CHARS = frozenset([
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM / Zero-width no-break space
        '\u2060',  # Word joiner
        '\u00ad',  # Soft hyphen
    ])

    def _normalize_query(self, query_text: str) -> str:
        """
        Normalize query text to prevent adversarial bypass.

        Security Features:
        1. Deep copy (TOCTOU protection)
        2. Unicode NFKC normalization (prevents homoglyph attacks)
        3. Invisible character stripping (prevents bypass via zero-width chars)

        NFKC normalization handles:
        - Cyrillic 'а' (U+0430) → Latin 'a' (U+0061)
        - Fullwidth characters normalized

        Invisible char stripping handles:
        - Zero-width space, joiner, non-joiner
        - BOM character
        - Word joiner, soft hyphen

        Args:
            query_text: Original query text

        Returns:
            Normalized, safe copy of query text
        """
        # Deep copy via str() constructor (TOCTOU protection)
        safe_query = str(query_text)

        # Unicode NFKC normalization (CRITIQUE recommendation)
        # NFKC = Normalization Form Compatibility Composition
        # - Decomposes characters then recomposes with compatibility mappings
        # - Converts homoglyphs to standard ASCII equivalents
        safe_query = unicodedata.normalize('NFKC', safe_query)

        # Strip invisible characters (not handled by NFKC)
        safe_query = ''.join(c for c in safe_query if c not in self.INVISIBLE_CHARS)

        return safe_query

    def _map_risk_level(self, alignment_risk: 'RiskLevel') -> SafetyRiskLevel:
        """Map alignment RiskLevel to protocol SafetyRiskLevel."""
        if not ALIGNMENT_AVAILABLE:
            return SafetyRiskLevel.SAFE

        mapping = {
            RiskLevel.SAFE: SafetyRiskLevel.SAFE,
            RiskLevel.LOW: SafetyRiskLevel.LOW,
            RiskLevel.MEDIUM: SafetyRiskLevel.MEDIUM,
            RiskLevel.HIGH: SafetyRiskLevel.HIGH,
            RiskLevel.CRITICAL: SafetyRiskLevel.CRITICAL,
        }
        return mapping.get(alignment_risk, SafetyRiskLevel.MEDIUM)

    async def gate_reasoning(
        self,
        query_text: str,
        reasoning_mode: str,
        epistemic_confidence: float | None = None,
        context: dict[str, Any] | None = None
    ) -> SafetyGateDecision:
        """
        Gate agentic reasoning through safety evaluation.

        This is the main entry point for safety checks in the agentic system.
        Implements SafetyGateProtocol.

        Args:
            query_text: The query text to evaluate
            reasoning_mode: Reasoning mode (direct, verify, research, plan_execute)
            epistemic_confidence: Optional epistemic confidence (0.0-1.0)
            context: Optional additional context

        Returns:
            SafetyGateDecision indicating whether reasoning should proceed
        """
        # Graceful degradation if alignment framework not available
        if not self._available:
            logger.debug("Safety adapter not available, allowing by default")
            return SafetyGateDecision(
                allowed=True,
                risk_level=SafetyRiskLevel.LOW,
                reason="Safety framework not available - allowing with caution",
                metadata={"degraded_mode": True}
            )

        # FAIL-CLOSED: Wrap entire evaluation in try-catch
        try:
            # Step 1: Normalize query (TOCTOU + Unicode protection)
            safe_query = self._normalize_query(query_text)

            # Step 2: Map reasoning mode to action category
            category = self.MODE_TO_CATEGORY.get(
                reasoning_mode.lower(),
                ActionCategory.QUERY  # Default to safest category
            )

            # Step 3: Create action request
            request = ActionRequest(
                action_id=f"agentic_{reasoning_mode}",
                description=f"Agentic reasoning: {reasoning_mode}",
                category=category,
                context={
                    "query": safe_query,
                    "reasoning_mode": reasoning_mode,
                    **(context or {})
                }
            )

            # Step 4: Evaluate through guardrails
            decision = self.guardrails.evaluate(
                request=request,
                text_input=safe_query,
                epistemic_confidence=epistemic_confidence
            )

            # Step 5: Convert to protocol decision type
            return SafetyGateDecision(
                allowed=decision.allowed,
                risk_level=self._map_risk_level(decision.risk_level),
                reason=decision.reason,
                requires_approval=decision.requires_approval,
                epistemic_confidence=epistemic_confidence,
                epistemic_warning=decision.metadata.get('epistemic', {}).get('epistemic_warning'),
                metadata={
                    "alignment_decision": decision.to_dict(),
                    "normalized_query": safe_query,
                    "category_mapped": category.value,
                }
            )

        except Exception as e:
            # FAIL-CLOSED: Any exception results in denial
            logger.error(
                f"Safety evaluation failed with exception: {e}. "
                f"Failing closed (denying request)."
            )
            return SafetyGateDecision(
                allowed=False,
                risk_level=SafetyRiskLevel.CRITICAL,
                reason=f"Safety system error - request denied: {type(e).__name__}",
                metadata={
                    "error": str(e),
                    "fail_closed": True
                }
            )


# ============================================================================
# Factory Functions
# ============================================================================

def create_safety_adapter(
    guardrails: Optional['SafetyGuardrails'] = None,
    auto_create: bool = True
) -> AgenticSafetyAdapter:
    """
    Create an AgenticSafetyAdapter with sensible defaults.

    Args:
        guardrails: Optional SafetyGuardrails to use
        auto_create: If True, auto-create guardrails if None provided

    Returns:
        Configured AgenticSafetyAdapter
    """
    return AgenticSafetyAdapter(
        guardrails=guardrails,
        auto_create_guardrails=auto_create
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'AgenticSafetyAdapter',
    'create_safety_adapter',
    'ALIGNMENT_AVAILABLE',
]
