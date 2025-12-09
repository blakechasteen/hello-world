"""
CoVe (Chain of Verification) + MRF (Metaprompting Refinement Framework) Integration

This module bridges Chain of Verification with the Metaprompting Refinement Framework,
enabling verification-based response enhancement as part of the MRF pipeline.

Features:
- CoVe as refinement strategy option
- Quality metrics extraction from verification results
- Automatic verification decision making
- Graceful degradation if verification module unavailable
- Complete provenance tracking

Philosophy: "Verify before refining, refine after verifying"
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

# Graceful imports with fallback
try:
    from HoloLoom.verification import (
        VerificationChain,
        VerificationResult,
        VerificationStatus,
        DegradationLevel,
        create_verification_chain,
    )
    VERIFICATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logger.warning("Verification module not available. CoVe integration disabled.")
    VERIFICATION_AVAILABLE = False
    VerificationChain = None
    VerificationResult = None
    VerificationStatus = None
    DegradationLevel = None
    create_verification_chain = None


class CoVeStrategyType(Enum):
    """CoVe as a refinement strategy option."""
    COVE = "cove"
    COVE_LIGHT = "cove_light"  # Faster verification, lower degradation
    COVE_FULL = "cove_full"  # Complete verification, full degradation


@dataclass
class CoVeEnhancedResult:
    """Result from CoVe + MRF enhancement."""

    original_response: str
    verified_response: str
    verification_result: Optional['VerificationResult']
    quality_improvement: float  # 0.0-1.0, improvement ratio
    verification_time_ms: float
    claims_extracted: int
    contradictions_found: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of verification results."""
        summary_parts = [
            f"Quality Improvement: +{self.quality_improvement:.1%}",
            f"Claims Verified: {self.claims_extracted}",
            f"Contradictions Found: {self.contradictions_found}",
            f"Verification Time: {self.verification_time_ms:.1f}ms"
        ]
        return " | ".join(summary_parts)


class VerificationMRFBridge:
    """
    Bridge between CoVe (verification) and MRF (metaprompting refinement).

    Coordinates verification and refinement for optimal response quality.
    """

    def __init__(
        self,
        verification_chain: Optional[Any] = None,
        degradation_level: Optional[Any] = None,
    ):
        """
        Initialize verification bridge.

        Args:
            verification_chain: Optional pre-configured VerificationChain
            degradation_level: Verification depth (PARTIAL, FULL)
        """
        if not VERIFICATION_AVAILABLE:
            logger.warning("Verification module not available. Bridge disabled.")
            self.available = False
            self.verification_chain = None
            self.degradation_level = None
            return

        self.available = True
        self.verification_chain = verification_chain
        self.degradation_level = degradation_level or DegradationLevel.PARTIAL

        # Create default chain if not provided
        if self.verification_chain is None:
            try:
                self.verification_chain = create_verification_chain(
                    degradation_level=self.degradation_level
                )
            except Exception as e:
                logger.error(f"Failed to create verification chain: {e}")
                self.available = False

    async def enhance_with_cove(
        self,
        query: str,
        response: str,
        confidence: float = 0.5,
    ) -> CoVeEnhancedResult:
        """
        Enhance response using CoVe verification.

        Args:
            query: Original user query
            response: Generated response to verify
            confidence: Initial confidence in response (0.0-1.0)

        Returns:
            CoVeEnhancedResult with verification details
        """
        if not self.available:
            logger.warning("Verification not available. Returning original response.")
            return CoVeEnhancedResult(
                original_response=response,
                verified_response=response,
                verification_result=None,
                quality_improvement=0.0,
                verification_time_ms=0.0,
                claims_extracted=0,
                contradictions_found=0,
            )

        # Decide if verification needed
        if not self.should_verify(confidence, threshold=0.75):
            logger.debug(f"Confidence {confidence:.2f} above threshold. Skipping verification.")
            return CoVeEnhancedResult(
                original_response=response,
                verified_response=response,
                verification_result=None,
                quality_improvement=0.0,
                verification_time_ms=0.0,
                claims_extracted=0,
                contradictions_found=0,
                metadata={"skipped_reason": "high_confidence"},
            )

        # Run verification
        start_time = time.time()

        try:
            verification_result = await self.verification_chain.verify(
                query=query,
                response=response,
            )
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return CoVeEnhancedResult(
                original_response=response,
                verified_response=response,
                verification_result=None,
                quality_improvement=0.0,
                verification_time_ms=(time.time() - start_time) * 1000,
                claims_extracted=0,
                contradictions_found=0,
                metadata={"error": str(e)},
            )

        verification_time_ms = (time.time() - start_time) * 1000

        # Extract quality signals
        quality_signals = self.get_quality_signals(verification_result)

        # Calculate quality improvement
        # Based on verified status, claims extracted, contradictions resolved
        quality_improvement = self._calculate_quality_improvement(
            quality_signals,
            verification_result,
        )

        # Get verified response (use corrected if available, otherwise original)
        verified_response = (
            verification_result.corrected_response
            if hasattr(verification_result, "corrected_response")
            and verification_result.corrected_response
            else response
        )

        # Extract claim and contradiction counts
        claims_count = len(verification_result.claims) if hasattr(verification_result, "claims") else 0
        contradictions_count = (
            len(verification_result.contradictions)
            if hasattr(verification_result, "contradictions")
            else 0
        )

        return CoVeEnhancedResult(
            original_response=response,
            verified_response=verified_response,
            verification_result=verification_result,
            quality_improvement=quality_improvement,
            verification_time_ms=verification_time_ms,
            claims_extracted=claims_count,
            contradictions_found=contradictions_count,
            metadata=quality_signals,
        )

    def get_quality_signals(
        self,
        result: Optional[Any],
    ) -> Dict[str, float]:
        """
        Extract quality metrics from verification result.

        Args:
            result: VerificationResult from CoVe chain

        Returns:
            Dictionary of quality metrics (0.0-1.0)
        """
        if result is None or not VERIFICATION_AVAILABLE:
            return {}

        signals = {}

        # Status-based signal
        if hasattr(result, "status"):
            if result.status == VerificationStatus.VERIFIED:
                signals["status_score"] = 1.0
            elif result.status == VerificationStatus.PARTIALLY_VERIFIED:
                signals["status_score"] = 0.6
            elif result.status == VerificationStatus.CONTRADICTED:
                signals["status_score"] = 0.2
            else:
                signals["status_score"] = 0.5

        # Contradiction penalty
        if hasattr(result, "contradictions"):
            contradiction_count = len(result.contradictions)
            signals["contradiction_penalty"] = max(0.0, 1.0 - (contradiction_count * 0.1))

        # Claim coverage
        if hasattr(result, "claims"):
            verified_claims = sum(
                1 for claim in result.claims
                if hasattr(claim, "verified") and claim.verified
            )
            total_claims = len(result.claims)
            if total_claims > 0:
                signals["claim_coverage"] = verified_claims / total_claims

        # Confidence (if available)
        if hasattr(result, "confidence"):
            signals["verification_confidence"] = result.confidence

        return signals

    def should_verify(
        self,
        confidence: float,
        threshold: float = 0.75,
    ) -> bool:
        """
        Decide if response should be verified.

        Args:
            confidence: Current confidence in response (0.0-1.0)
            threshold: Confidence threshold for verification (default: 0.75)

        Returns:
            True if verification should be run, False otherwise
        """
        if not self.available:
            return False

        # Verify if confidence is below threshold
        return confidence < threshold

    def _calculate_quality_improvement(
        self,
        quality_signals: Dict[str, float],
        result: Optional[Any],
    ) -> float:
        """
        Calculate quality improvement percentage.

        Args:
            quality_signals: Quality metrics from verification
            result: VerificationResult

        Returns:
            Quality improvement (0.0-1.0)
        """
        if not quality_signals:
            return 0.0

        # Weighted combination of quality signals
        improvement = 0.0

        if "status_score" in quality_signals:
            improvement += quality_signals["status_score"] * 0.4

        if "contradiction_penalty" in quality_signals:
            improvement += quality_signals["contradiction_penalty"] * 0.3

        if "claim_coverage" in quality_signals:
            improvement += quality_signals["claim_coverage"] * 0.2

        if "verification_confidence" in quality_signals:
            improvement += quality_signals["verification_confidence"] * 0.1

        return min(1.0, max(0.0, improvement))


class CoVeRefinementStrategy:
    """
    CoVe as a refinement strategy for MRF.

    Implements strategy pattern for integration with MRF's refinement engine.
    """

    def __init__(
        self,
        bridge: Optional[VerificationMRFBridge] = None,
        degradation_level: Optional[Any] = None,
    ):
        """
        Initialize CoVe refinement strategy.

        Args:
            bridge: Optional pre-configured VerificationMRFBridge
            degradation_level: Verification depth (PARTIAL, FULL)
        """
        self.bridge = bridge
        if self.bridge is None and VERIFICATION_AVAILABLE:
            self.bridge = VerificationMRFBridge(degradation_level=degradation_level)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "cove"

    async def refine(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Apply CoVe verification as refinement strategy.

        Args:
            query: Original query
            response: Response to refine
            context: Additional context (e.g., confidence)

        Returns:
            Verified/refined response
        """
        if not self.bridge or not self.bridge.available:
            logger.warning("CoVe bridge not available. Returning original response.")
            return response

        context = context or {}
        confidence = context.get("confidence", 0.5)

        # Apply CoVe enhancement
        result = await self.bridge.enhance_with_cove(
            query=query,
            response=response,
            confidence=confidence,
        )

        logger.debug(f"CoVe refinement: {result.summary()}")

        # Return verified response
        return result.verified_response

    def can_apply(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Return applicability score for this strategy.

        Args:
            context: Context information

        Returns:
            Applicability score (0.0-1.0)
        """
        if not self.bridge or not self.bridge.available:
            return 0.0

        context = context or {}
        confidence = context.get("confidence", 0.5)

        # More applicable when confidence is low
        applicability = 1.0 - confidence

        # Check if claim-heavy query
        if context.get("query_type") == "factual":
            applicability += 0.2

        return min(1.0, applicability)


# Factory Functions

def create_cove_strategy(
    degradation_level: Optional[Any] = None,
) -> Optional[CoVeRefinementStrategy]:
    """
    Create CoVe refinement strategy.

    Args:
        degradation_level: Verification depth (PARTIAL, FULL)

    Returns:
        CoVeRefinementStrategy or None if verification unavailable
    """
    if not VERIFICATION_AVAILABLE:
        logger.warning("Verification module not available. Cannot create CoVe strategy.")
        return None

    return CoVeRefinementStrategy(degradation_level=degradation_level)


def create_verification_bridge(
    degradation_level: Optional[Any] = None,
) -> Optional[VerificationMRFBridge]:
    """
    Create verification bridge.

    Args:
        degradation_level: Verification depth (PARTIAL, FULL)

    Returns:
        VerificationMRFBridge or None if verification unavailable
    """
    if not VERIFICATION_AVAILABLE:
        logger.warning("Verification module not available. Cannot create bridge.")
        return None

    return VerificationMRFBridge(degradation_level=degradation_level)


def enable_cove_for_mrf(
    mrf_engine: Any,
    degradation_level: Optional[Any] = None,
) -> bool:
    """
    Register CoVe strategy with MRF engine.

    Args:
        mrf_engine: UnifiedMRF instance
        degradation_level: Verification depth

    Returns:
        True if registration successful, False otherwise
    """
    if not VERIFICATION_AVAILABLE:
        logger.warning("Verification module not available. Cannot enable CoVe for MRF.")
        return False

    try:
        strategy = create_cove_strategy(degradation_level=degradation_level)
        if strategy is None:
            return False

        # Register strategy with MRF (assumes MRF has register_strategy method)
        if hasattr(mrf_engine, "register_strategy"):
            mrf_engine.register_strategy(CoVeStrategyType.COVE.value, strategy)
            logger.info("CoVe strategy registered with MRF engine")
            return True
        else:
            logger.warning("MRF engine does not support strategy registration")
            return False

    except Exception as e:
        logger.error(f"Failed to enable CoVe for MRF: {e}")
        return False


# Convenience function for MRF integration

async def verify_and_refine(
    query: str,
    response: str,
    confidence: float = 0.5,
    degradation_level: Optional[Any] = None,
) -> CoVeEnhancedResult:
    """
    Convenience function: verify response and refine using CoVe.

    Single-call verification + refinement workflow.

    Args:
        query: Original query
        response: Generated response
        confidence: Initial confidence in response
        degradation_level: Verification depth

    Returns:
        CoVeEnhancedResult with verification details
    """
    bridge = create_verification_bridge(degradation_level=degradation_level)
    if bridge is None:
        return CoVeEnhancedResult(
            original_response=response,
            verified_response=response,
            verification_result=None,
            quality_improvement=0.0,
            verification_time_ms=0.0,
            claims_extracted=0,
            contradictions_found=0,
        )

    return await bridge.enhance_with_cove(
        query=query,
        response=response,
        confidence=confidence,
    )
