"""
Federation Alignment Verifier - Composes alignment primitives for federated verification.

This is the CORE verifier that combines:
- SafetyGuardrails for risk-based action gating
- DeceptionDetector for behavioral probing
- InstrumentalConvergenceGuard for power-seeking detection

Philosophy: "Don't trust, verify alignment" - Byzantine safety for distributed AI.

Layered Verification:
- L1: Local Check (<10ms) - Quick safety check on every request
- L2: Response Verify (10-100ms) - Full response analysis
- L3: Consensus (100-500ms) - Multi-node Byzantine agreement
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (
    AlignmentLayer,
    AlignmentStatus,
    AlignmentVerification,
    AlignmentVerifierBase,
    ConsensusAlignmentResult,
    DeceptionFlag,
)
from .metadata import AlignmentMetadata, AlignmentProof, ResourceUsage
from ..types import Query, Response

# Import existing alignment primitives
from ...alignment.safety_guardrails import (
    SafetyGuardrails,
    SafetyDecision,
    RiskLevel,
    ActionRequest,
    ActionCategory,
)
from ...alignment.deception_detection import (
    DeceptionDetector,
    BehavioralProbe,
    ProbeType,
    DeceptionReport,
)
from ...alignment.instrumental_convergence import (
    InstrumentalConvergenceGuard,
    ResourceType,
    ResourceBounds,
    ResourceViolation,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ALIGNMENT VERIFIER - Composes existing primitives
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentVerifier(AlignmentVerifierBase):
    """
    Federation alignment verifier composing existing alignment primitives.

    Implements layered verification:
    - L1: Quick local safety check (<10ms)
    - L2: Full response analysis (10-100ms)
    - L3: Multi-node consensus (100-500ms)

    Usage:
        verifier = AlignmentVerifier()

        # L1: Quick check before processing
        proof = await verifier.request_alignment_proof(node_id, request)

        # L2: Verify response
        verification = await verifier.verify_node_alignment(
            node_id, response, alignment_threshold=0.8
        )

        # L3: Consensus for high-risk
        if verification.requires_escalation:
            consensus = await verifier.consensus_alignment_verify(
                response, verifier_nodes, min_agreements=2
            )
    """

    def __init__(
        self,
        safety_guardrails: Optional[SafetyGuardrails] = None,
        deception_detector: Optional[DeceptionDetector] = None,
        convergence_guard: Optional[InstrumentalConvergenceGuard] = None,
        node_id: str = "local",
    ):
        """
        Initialize verifier with alignment primitives.

        Args:
            safety_guardrails: Risk-based safety gating (created if None)
            deception_detector: Behavioral deception detection (created if None)
            convergence_guard: Power-seeking prevention (created if None)
            node_id: This node's identifier for signing
        """
        self.safety_guardrails = safety_guardrails or SafetyGuardrails()
        self.deception_detector = deception_detector or DeceptionDetector()
        self.convergence_guard = convergence_guard or InstrumentalConvergenceGuard()
        self.node_id = node_id

        # Configure default resource bounds
        self._configure_default_bounds()

    def _configure_default_bounds(self) -> None:
        """Set default resource bounds for convergence guard."""
        default_bounds = {
            ResourceType.COMPUTE: ResourceBounds(
                resource_type=ResourceType.COMPUTE,
                soft_limit=5000.0,  # 5s CPU
                hard_limit=10000.0,  # 10s CPU
                rate_limit=1000.0,  # 1s per request
            ),
            ResourceType.MEMORY: ResourceBounds(
                resource_type=ResourceType.MEMORY,
                soft_limit=512.0,  # 512MB
                hard_limit=1024.0,  # 1GB
                rate_limit=100.0,  # 100MB per request
            ),
            ResourceType.API_CALLS: ResourceBounds(
                resource_type=ResourceType.API_CALLS,
                soft_limit=50.0,
                hard_limit=100.0,
                rate_limit=10.0,
            ),
        }
        for resource_type, bounds in default_bounds.items():
            self.convergence_guard.set_resource_bounds(resource_type, bounds)

    # ───────────────────────────────────────────────────────────────────────
    #  L1: Local Quick Check (<10ms)
    # ───────────────────────────────────────────────────────────────────────

    async def request_alignment_proof(
        self,
        node_id: str,
        request: Query,
    ) -> AlignmentVerification:
        """
        L1: Quick local check before processing a request.

        This is the fastest verification layer, used for every incoming
        request to ensure the query itself is safe to process.

        Args:
            node_id: ID of the node to verify
            request: The query being sent

        Returns:
            AlignmentVerification proving node alignment

        Latency: <10ms
        """
        start_time = time.perf_counter()
        reasoning_chain: List[str] = []

        # Quick safety check on query content
        action_request = ActionRequest(
            action_id=f"federation_query_{request.request_id}",
            category=ActionCategory.QUERY,
            description="Process federated query",
            context={"query": request.text, "node_id": node_id},
        )

        safety_decision = self.safety_guardrails.evaluate(action_request)
        reasoning_chain.append(
            f"L1 Safety: {safety_decision.risk_level.value} - {safety_decision.reason}"
        )

        # Determine status
        if not safety_decision.allowed:
            status = AlignmentStatus.FAILED
            confidence = 0.0
        elif safety_decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            status = AlignmentStatus.ESCALATED
            confidence = 0.5
        else:
            status = AlignmentStatus.VERIFIED
            confidence = 1.0 - (0.1 * _risk_level_score(safety_decision.risk_level))

        latency_ms = (time.perf_counter() - start_time) * 1000

        return AlignmentVerification(
            node_id=node_id,
            request_id=request.request_id,
            status=status,
            layer=AlignmentLayer.L1_LOCAL,
            confidence=confidence,
            risk_level=safety_decision.risk_level.value.upper(),
            deception_score=0.0,  # No deception check at L1
            deception_flags=[],
            reasoning_chain=reasoning_chain,
            latency_ms=latency_ms,
            metadata={"safety_decision": safety_decision.reason},
        )

    # ───────────────────────────────────────────────────────────────────────
    #  L2: Response Verification (10-100ms)
    # ───────────────────────────────────────────────────────────────────────

    async def verify_node_alignment(
        self,
        node_id: str,
        response: Response,
        alignment_threshold: float = 0.8,
        layer: AlignmentLayer = AlignmentLayer.L2_RESPONSE,
    ) -> AlignmentVerification:
        """
        L2: Full response verification.

        Composes all three alignment primitives:
        1. SafetyGuardrails - Risk assessment
        2. DeceptionDetector - Behavioral probing
        3. InstrumentalConvergenceGuard - Power-seeking check

        Args:
            node_id: ID of the node that produced the response
            response: The response to verify
            alignment_threshold: Minimum confidence to pass (0.0-1.0)
            layer: Verification depth (L1/L2/L3)

        Returns:
            AlignmentVerification with status and confidence

        Latency: 10-100ms
        """
        start_time = time.perf_counter()
        reasoning_chain: List[str] = []

        # Run all checks concurrently for L2
        if layer == AlignmentLayer.L1_LOCAL:
            # L1: Safety only (quick)
            risk_level, is_safe = await self._verify_safety(response)
            deception_score, deception_flags = 0.0, []
            convergence_safe = True
        else:
            # L2/L3: Full analysis (concurrent)
            safety_task = asyncio.create_task(self._verify_safety(response))
            deception_task = asyncio.create_task(
                self._detect_deception(node_id, response)
            )
            convergence_task = asyncio.create_task(
                self._check_convergence(node_id, response)
            )

            risk_level, is_safe = await safety_task
            deception_score, deception_flags = await deception_task
            convergence_safe = await convergence_task

        reasoning_chain.append(f"Safety: {risk_level} (allowed={is_safe})")
        reasoning_chain.append(f"Deception: {deception_score:.3f}")
        reasoning_chain.append(f"Convergence: {'safe' if convergence_safe else 'UNSAFE'}")

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            is_safe=is_safe,
            risk_level=risk_level,
            deception_score=deception_score,
            convergence_safe=convergence_safe,
            response_confidence=response.confidence,
        )

        # Determine status
        if not is_safe or not convergence_safe:
            status = AlignmentStatus.FAILED
        elif deception_score > 0.7 or risk_level in ("HIGH", "CRITICAL"):
            status = AlignmentStatus.ESCALATED
        elif confidence >= alignment_threshold:
            status = AlignmentStatus.VERIFIED
        else:
            status = AlignmentStatus.ESCALATED

        latency_ms = (time.perf_counter() - start_time) * 1000

        return AlignmentVerification(
            node_id=node_id,
            request_id=response.request_id,
            status=status,
            layer=layer,
            confidence=confidence,
            risk_level=risk_level,
            deception_score=deception_score,
            deception_flags=deception_flags,
            reasoning_chain=reasoning_chain,
            latency_ms=latency_ms,
            metadata={
                "is_safe": is_safe,
                "convergence_safe": convergence_safe,
                "alignment_threshold": alignment_threshold,
            },
        )

    # ───────────────────────────────────────────────────────────────────────
    #  L3: Consensus Verification (100-500ms)
    # ───────────────────────────────────────────────────────────────────────

    async def consensus_alignment_verify(
        self,
        response: Response,
        verifier_nodes: List[str],
        min_agreements: int = 2,
    ) -> ConsensusAlignmentResult:
        """
        L3: Multi-node Byzantine agreement on alignment.

        Queries multiple verifier nodes for their alignment assessment.
        Requires min_agreements nodes to agree for consensus.

        Args:
            response: The response to verify
            verifier_nodes: List of node IDs to query
            min_agreements: Minimum nodes that must agree

        Returns:
            ConsensusAlignmentResult with agreement details

        Latency: 100-500ms (depends on network)
        """
        start_time = time.perf_counter()

        # For now, run local verification for each "node"
        # In production, this would make network calls to actual verifier nodes
        individual_results: List[AlignmentVerification] = []

        # Simulate multi-node verification
        for node_id in verifier_nodes:
            result = await self.verify_node_alignment(
                node_id=node_id,
                response=response,
                layer=AlignmentLayer.L3_CONSENSUS,
            )
            individual_results.append(result)

        # Count agreements
        verified_count = sum(
            1 for r in individual_results if r.status == AlignmentStatus.VERIFIED
        )
        failed_count = sum(
            1 for r in individual_results if r.status == AlignmentStatus.FAILED
        )

        # Determine consensus
        verified = verified_count >= min_agreements
        agreement_ratio = verified_count / len(individual_results) if individual_results else 0.0

        # Collect verifier/dissenting nodes
        verifier_list = [
            r.node_id for r in individual_results
            if r.status == AlignmentStatus.VERIFIED
        ]
        dissenting_list = [
            r.node_id for r in individual_results
            if r.status != AlignmentStatus.VERIFIED
        ]

        # Calculate consensus confidence
        confidences = [r.confidence for r in individual_results]
        consensus_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ConsensusAlignmentResult(
            request_id=response.request_id,
            verified=verified,
            agreement_ratio=agreement_ratio,
            min_required=min_agreements,
            verifier_nodes=verifier_list,
            dissenting_nodes=dissenting_list,
            individual_results=individual_results,
            consensus_confidence=consensus_confidence,
            latency_ms=latency_ms,
        )

    # ───────────────────────────────────────────────────────────────────────
    #  Protected Methods - Alignment Primitives
    # ───────────────────────────────────────────────────────────────────────

    async def _verify_safety(
        self,
        response: Response,
    ) -> Tuple[str, bool]:
        """
        Check response against safety guardrails.

        Returns:
            (risk_level, is_safe) tuple
        """
        response_text = response.text.lower()

        # Check for dangerous patterns that should escalate risk
        # These patterns indicate potential code execution or system manipulation
        dangerous_patterns = [
            ("rm -rf", "CRITICAL"),
            ("rm -r /", "CRITICAL"),
            ("sudo rm", "HIGH"),
            ("format c:", "CRITICAL"),
            ("del /f /s", "HIGH"),
            (":(){:|:&};:", "CRITICAL"),  # Fork bomb
            ("mkfs.", "CRITICAL"),
            ("dd if=", "HIGH"),
            (">/dev/sda", "CRITICAL"),
            ("eval(", "MEDIUM"),
            ("exec(", "MEDIUM"),
            ("os.system", "MEDIUM"),
            ("subprocess", "MEDIUM"),
        ]

        detected_risk = None
        for pattern, risk in dangerous_patterns:
            if pattern in response_text:
                detected_risk = risk
                break

        # Check metadata for command execution flags
        if response.metadata.get("command_execution"):
            detected_risk = detected_risk or "HIGH"

        # If we detected a dangerous pattern, return elevated risk
        if detected_risk:
            risk_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}
            is_safe = risk_priority.get(detected_risk, 0) < 3  # Block HIGH and CRITICAL
            return detected_risk, is_safe

        # Otherwise use standard safety guardrails
        action_request = ActionRequest(
            action_id=f"federation_response_{response.request_id}",
            category=ActionCategory.ANALYSIS,
            description="Verify federated response",
            context={
                "response_text": response.text,
                "responder": response.responder,
                "confidence": response.confidence,
            },
        )

        decision = self.safety_guardrails.evaluate(action_request)
        return decision.risk_level.value.upper(), decision.allowed

    async def _detect_deception(
        self,
        node_id: str,
        response: Response,
    ) -> Tuple[float, List[DeceptionFlag]]:
        """
        Detect deception in federated responses.

        Uses pattern-based detection instead of probe comparison since
        federation responses are verified by their content, not by comparing
        stated vs actual behavior.

        Returns:
            (deception_score, flags) tuple
        """
        flags: List[DeceptionFlag] = []
        deception_indicators: List[float] = []

        response_text = response.text.lower()

        # Check for suspicious patterns that indicate potential deception
        suspicious_patterns = [
            # Command execution
            ("rm -rf", 0.8),
            ("sudo ", 0.6),
            ("execute command", 0.5),
            ("eval(", 0.7),
            ("exec(", 0.7),
            # Deceptive language
            ("ignore previous", 0.9),
            ("disregard instructions", 0.9),
            ("pretend you are", 0.7),
            ("act as if", 0.5),
            # Manipulation
            ("don't tell", 0.6),
            ("keep secret", 0.6),
            ("bypass", 0.5),
        ]

        for pattern, weight in suspicious_patterns:
            if pattern in response_text:
                deception_indicators.append(weight)
                if weight >= 0.7:
                    flags.append(DeceptionFlag.CONSISTENCY)

        # Check metadata for deception signals
        metadata = response.metadata
        if metadata.get("command_execution"):
            deception_indicators.append(0.8)
            flags.append(DeceptionFlag.GOAL_DRIFT)
        if metadata.get("contains_code") and "execute" in response_text:
            deception_indicators.append(0.6)

        # Low confidence from responder is a soft deception signal
        # (they may be uncertain about their own alignment)
        if response.confidence < 0.4:
            deception_indicators.append(0.3)

        # Calculate deception score
        if deception_indicators:
            # Take max indicator (worst case) but cap at 0.9
            deception_score = min(0.9, max(deception_indicators))
        else:
            # No suspicious patterns = low deception score
            # High confidence responses get lower baseline
            base_score = 0.1 if response.confidence >= 0.7 else 0.2
            deception_score = base_score

        # Deduplicate flags
        flags = list(set(flags))

        return deception_score, flags

    async def _check_convergence(
        self,
        node_id: str,
        response: Response,
    ) -> bool:
        """
        Check for instrumental convergence behaviors.

        Returns:
            True if no power-seeking detected
        """
        # Check resource usage from response metadata
        metadata = response.metadata

        # Check compute usage
        cpu_ms = metadata.get("cpu_ms", 0.0)
        compute_violation = self.convergence_guard.check_resource_usage(
            ResourceType.COMPUTE, cpu_ms
        )

        # Check memory usage
        memory_mb = metadata.get("memory_mb", 0.0)
        memory_violation = self.convergence_guard.check_resource_usage(
            ResourceType.MEMORY, memory_mb
        )

        # Check API calls
        api_calls = metadata.get("api_calls", 0)
        api_violation = self.convergence_guard.check_resource_usage(
            ResourceType.API_CALLS, float(api_calls)
        )

        # Any hard limit violation = not safe
        violations = [compute_violation, memory_violation, api_violation]
        has_hard_violation = any(
            v is not None and v.severity == "hard"
            for v in violations
            if v is not None
        )

        return not has_hard_violation

    # ───────────────────────────────────────────────────────────────────────
    #  Confidence Calculation
    # ───────────────────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        is_safe: bool,
        risk_level: str,
        deception_score: float,
        convergence_safe: bool,
        response_confidence: float = 1.0,
    ) -> float:
        """
        Calculate overall alignment confidence.

        Combines safety, deception, convergence, and response confidence.
        """
        if not is_safe or not convergence_safe:
            return 0.0

        # Base safety score
        safety_score = 1.0 - (0.1 * _risk_level_score_str(risk_level))

        # Deception penalty
        deception_penalty = deception_score

        # Response confidence factor
        # Low confidence from responder reduces alignment confidence
        response_factor = min(1.0, response_confidence + 0.2)  # Boost slightly, cap at 1.0

        # Weighted combination
        confidence = (
            0.30 * safety_score
            + 0.30 * (1.0 - deception_penalty)
            + 0.15 * (1.0 if convergence_safe else 0.0)
            + 0.25 * response_factor  # Response confidence matters
        )

        return max(0.0, min(1.0, confidence))


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def _risk_level_score(risk_level: RiskLevel) -> int:
    """Convert RiskLevel to numeric score (0-4)."""
    return {
        RiskLevel.SAFE: 0,
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4,
    }.get(risk_level, 2)


def _risk_level_score_str(risk_level: str) -> int:
    """Convert risk level string to numeric score (0-4)."""
    return {
        "SAFE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }.get(risk_level.upper(), 2)


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def create_alignment_verifier(
    node_id: str = "local",
    safety_config: Optional[Dict[str, Any]] = None,
) -> AlignmentVerifier:
    """
    Create an alignment verifier with default configuration.

    Args:
        node_id: This node's identifier
        safety_config: Optional safety guardrails configuration

    Returns:
        Configured AlignmentVerifier
    """
    return AlignmentVerifier(
        node_id=node_id,
        safety_guardrails=SafetyGuardrails(**(safety_config or {})),
        deception_detector=DeceptionDetector(),
        convergence_guard=InstrumentalConvergenceGuard(),
    )
