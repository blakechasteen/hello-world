#!/usr/bin/env python3
"""
Demo: Chain of Verification (CoVe) System

Demonstrates HoloLoom's implementation of Meta AI's Chain-of-Verification
methodology for reducing hallucinations through systematic verification.

4-Step CoVe Process:
1. Baseline Response - Generate initial answer
2. Verification Planning - Create targeted fact-checking questions
3. Independent Verification - Answer questions WITHOUT seeing original
4. Refined Response - Generate final answer using verified facts

Key Innovation: Independent verification prevents self-confirmation bias.

Created: 2025-12-05
"""

import asyncio
import sys
sys.path.insert(0, '.')

from hololoom.verification import (
    # Main chain
    VerificationChain,
    verify_response,
    create_verification_chain,
    # Component factories
    create_claim_extractor,
    create_verification_planner,
    create_independent_verifier,
    create_contradiction_detector,
    create_result_synthesizer,
    # Configuration
    VerificationConfig,
    DegradationLevel,
    # Types
    VerificationStatus,
    ClaimType,
)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_result(result):
    """Print verification result summary."""
    print(result.summary())
    print()

    if result.claims_extracted:
        print(f"Claims extracted ({len(result.claims_extracted)}):")
        for i, claim in enumerate(result.claims_extracted[:3], 1):
            print(f"  {i}. [{claim.claim_type.value}] {claim.text[:60]}...")

    if result.verification_questions:
        print(f"\nVerification questions ({len(result.verification_questions)}):")
        for i, q in enumerate(result.verification_questions[:3], 1):
            print(f"  {i}. {q.question}")

    if result.contradictions_found:
        print(f"\nContradictions found ({len(result.contradictions_found)}):")
        for i, c in enumerate(result.contradictions_found, 1):
            print(f"  {i}. [{c.contradiction_type}] {c.explanation[:60]}...")


async def demo_basic_verification():
    """Demo 1: Basic verification chain."""
    print_header("Demo 1: Basic Chain of Verification")

    # Create verification chain
    chain = create_verification_chain()

    # Sample response to verify
    query = "What is Thompson Sampling?"
    response = """
    Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
    It was introduced in 1933 by William R. Thompson.
    The algorithm balances exploration and exploitation by maintaining a
    probability distribution over each arm's expected reward.
    At each step, it samples from these distributions and selects the
    arm with the highest sampled value.
    """

    print(f"Query: {query}")
    print(f"Response to verify: {response[:100]}...")
    print()

    # Verify the response
    result = await chain.verify(
        query=query,
        response=response,
        confidence=0.7,  # Initial confidence
    )

    print_result(result)


async def demo_contradiction_detection():
    """Demo 2: Detecting contradictions in responses."""
    print_header("Demo 2: Contradiction Detection")

    chain = create_verification_chain()

    # Response with factual errors
    query = "When was Thompson Sampling introduced?"
    response = """
    Thompson Sampling was introduced in 1950 by John Thompson.
    It is a frequentist approach to decision making under uncertainty.
    The algorithm always selects the option with the highest expected value.
    """

    print(f"Query: {query}")
    print(f"Response with errors: {response[:100]}...")
    print()

    result = await chain.verify(
        query=query,
        response=response,
        confidence=0.6,
    )

    print_result(result)

    # Show refined response if available
    if result.refinement_applied:
        print("\nRefined response:")
        print(f"  {result.verified_response[:200]}...")


async def demo_claim_extraction():
    """Demo 3: Extracting verifiable claims from text."""
    print_header("Demo 3: Claim Extraction")

    config = VerificationConfig()
    extractor = create_claim_extractor(config, llm_client=None)

    text = """
    Machine learning models can process images at 1000 frames per second.
    Deep learning was invented in 2012 with AlexNet.
    Neural networks contain multiple layers of interconnected nodes.
    The backpropagation algorithm is used to train neural networks.
    Training GPT-4 cost approximately $100 million.
    """

    print(f"Text to analyze: {text[:100]}...")
    print()

    claims = await extractor.extract(text, {})

    print(f"Extracted {len(claims)} verifiable claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\n  Claim {i}:")
        print(f"    Type: {claim.claim_type.value}")
        print(f"    Text: {claim.text}")
        print(f"    Importance: {claim.importance:.2f}")
        if claim.entities:
            print(f"    Entities: {', '.join(claim.entities[:3])}")


async def demo_degradation_levels():
    """Demo 4: Graceful degradation levels."""
    print_header("Demo 4: Degradation Levels")

    levels = [
        (DegradationLevel.FULL, "Full LLM-based verification"),
        (DegradationLevel.PARTIAL, "Local LLM (Ollama)"),
        (DegradationLevel.HEURISTIC, "Rule-based only"),
        (DegradationLevel.DISABLED, "Verification disabled"),
    ]

    query = "What is reinforcement learning?"
    response = "Reinforcement learning is a type of machine learning."

    for level, description in levels:
        config = VerificationConfig(preferred_level=level)
        chain = create_verification_chain(config=config)

        result = await chain.verify(
            query=query,
            response=response,
            confidence=0.8,
        )

        print(f"{level.value.upper()}: {description}")
        print(f"  Status: {result.status.value}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print(f"  Claims: {len(result.claims_extracted)}")
        print()


async def demo_quick_verify():
    """Demo 5: Quick verification using convenience function."""
    print_header("Demo 5: Quick Verification API")

    result = await verify_response(
        query="What is attention mechanism?",
        response="Attention allows models to focus on relevant parts of the input.",
        confidence=0.75,
    )

    print("Using verify_response() convenience function:")
    print_result(result)


async def demo_iterative_refinement():
    """Demo 6: Iterative refinement for low-confidence responses."""
    print_header("Demo 6: Iterative Refinement")

    config = VerificationConfig(
        max_iterations=3,
        confidence_threshold=0.85,
    )
    chain = create_verification_chain(config=config)

    # Low-confidence response needing refinement
    result = await chain.verify(
        query="Explain gradient descent",
        response="Gradient descent optimizes functions by following gradients.",
        confidence=0.5,  # Low initial confidence
        max_iterations=3,
    )

    print(f"Iterations performed: {result.iterations}")
    print(f"Confidence: {result.confidence_before:.2f} -> {result.confidence_after:.2f}")
    print(f"Refinement applied: {result.refinement_applied}")
    print()
    print_result(result)


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  Chain of Verification (CoVe) - Demo Suite")
    print("  Following Meta AI's 4-step methodology")
    print("=" * 60)

    await demo_basic_verification()
    await demo_contradiction_detection()
    await demo_claim_extraction()
    await demo_degradation_levels()
    await demo_quick_verify()
    await demo_iterative_refinement()

    print_header("Summary")
    print("The Chain of Verification system provides:")
    print("  1. Automatic claim extraction from responses")
    print("  2. Independent verification (prevents self-confirmation bias)")
    print("  3. Semantic contradiction detection")
    print("  4. Graceful degradation (FULL -> PARTIAL -> HEURISTIC)")
    print("  5. Iterative refinement for low-confidence responses")
    print("  6. Complete provenance tracking")
    print()
    print("For more details, see:")
    print("  - HoloLoom/verification/README.md")
    print("  - CLAUDE.md (Chain of Verification section)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
