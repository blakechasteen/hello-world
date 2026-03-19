"""
Chain of Verification (CoVe) Module.

Implements Meta AI's 4-step CoVe methodology for reducing hallucinations:
1. Baseline Response - Generate initial answer
2. Verification Planning - Create targeted fact-checking questions
3. Independent Verification - Answer questions WITHOUT seeing original
4. Refined Response - Generate final answer using verified facts

Key Innovation: Independent verification prevents self-confirmation bias.

Usage:
    # Simple verification
    from hololoom.verification import verify_response

    result = await verify_response(
        query="What is Thompson Sampling?",
        response="Thompson Sampling is a Bayesian approach...",
        confidence=0.7
    )

    # With chain for repeated verifications
    from hololoom.verification import create_verification_chain

    chain = create_verification_chain(llm_client=my_llm)
    result = await chain.verify(query, response, confidence)

Created: 2025-12-05
"""

# Core protocols and types
# Main chain
from hololoom.verification.chain import (
    AgenticVerificationIntegration,
    VerificationCache,
    VerificationChain,
    VerificationMetrics,
    create_verification_chain,
    # Convenience functions
    verify_response,
)

# Component factories
from hololoom.verification.claim_extractor import create_claim_extractor
from hololoom.verification.contradiction_detector import create_contradiction_detector
from hololoom.verification.independent_verifier import create_independent_verifier
from hololoom.verification.protocol import (
    # Protocols
    ClaimExtractorProtocol,
    # Enums
    ClaimType,
    # Data classes
    Contradiction,
    ContradictionDetectorProtocol,
    ContradictionType,
    DegradationLevel,
    IndependentVerifierProtocol,
    ResultSynthesizerProtocol,
    VerifiableClaim,
    VerificationAnswer,
    VerificationChainProtocol,
    VerificationConfig,
    VerificationPlannerProtocol,
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
)
from hololoom.verification.result_synthesizer import create_result_synthesizer
from hololoom.verification.verification_planner import create_verification_planner

__all__ = [
    # Enums
    'ClaimType',
    'ContradictionType',
    'DegradationLevel',
    'VerificationStatus',
    # Data classes
    'Contradiction',
    'VerifiableClaim',
    'VerificationAnswer',
    'VerificationConfig',
    'VerificationQuestion',
    'VerificationResult',
    # Protocols
    'ClaimExtractorProtocol',
    'ContradictionDetectorProtocol',
    'IndependentVerifierProtocol',
    'ResultSynthesizerProtocol',
    'VerificationChainProtocol',
    'VerificationPlannerProtocol',
    # Main classes
    'VerificationChain',
    'VerificationCache',
    'VerificationMetrics',
    'AgenticVerificationIntegration',
    # Convenience functions
    'verify_response',
    'create_verification_chain',
    # Component factories
    'create_claim_extractor',
    'create_verification_planner',
    'create_independent_verifier',
    'create_contradiction_detector',
    'create_result_synthesizer',
]
