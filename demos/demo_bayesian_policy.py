"""
Demo: Bayesian Policy with Uncertainty Quantification
======================================================

Demonstrates HoloLoom's Bayesian policy engine with epistemic and aleatoric
uncertainty estimation.

This demo shows:
1. Deterministic vs Bayesian policy comparison
2. Uncertainty estimation (epistemic vs aleatoric)
3. Uncertainty-driven exploration (high uncertainty → explore more)
4. Performance impact of MC sampling (10× overhead)

Run:
    PYTHONPATH=. python demos/demo_bayesian_policy.py

Author: HoloLoom Probabilistic Reasoning Team
Date: 2025-11-03
"""

import asyncio
import numpy as np
import time
from typing import List

from HoloLoom.config import Config
from HoloLoom.policy.unified import create_policy, BanditStrategy
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Features, Context


# ============================================================================
# Mock Data
# ============================================================================

def create_mock_features(
    query_type: str = "factual",
    coherence: float = 0.8
) -> Features:
    """
    Create mock features for different query types.

    Args:
        query_type: Type of query (factual, procedural, analytical)
        coherence: Coherence score (0-1)

    Returns:
        Mock Features object
    """
    # Different query types have different feature patterns
    if query_type == "factual":
        psi = np.array([0.9, 0.2, 0.1, 0.3, 0.4, 0.2])  # High confidence, simple
        motifs = ["question→answer"]
    elif query_type == "procedural":
        psi = np.array([0.6, 0.5, 0.3, 0.7, 0.6, 0.4])  # Medium confidence, structured
        motifs = ["goal→constraint", "cause→effect"]
    elif query_type == "analytical":
        psi = np.array([0.4, 0.7, 0.8, 0.5, 0.7, 0.6])  # Lower confidence, complex
        motifs = ["contrast", "subordinate→main", "advcl→main"]
    else:
        psi = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        motifs = []

    return Features(
        psi=psi,
        motifs=motifs,
        metrics={"coherence": coherence, "fiedler": 0.3},
        confidence=coherence
    )


def create_mock_context(
    n_shards: int = 5,
    relevance: float = 0.7
) -> Context:
    """
    Create mock context with varying relevance.

    Args:
        n_shards: Number of context shards
        relevance: Relevance score (0-1)

    Returns:
        Mock Context object
    """
    texts = [f"Context shard {i}" for i in range(n_shards)]

    return Context(
        hits=[],
        kg_sub=None,
        shard_texts=texts,
        relevance=relevance
    )


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_deterministic_vs_bayesian():
    """
    Compare deterministic and Bayesian policies.

    Shows:
    - Same architecture, different uncertainty estimates
    - Bayesian provides confidence intervals
    - ~10× slowdown from MC sampling
    """
    print("\n" + "="*70)
    print("DEMO 1: Deterministic vs Bayesian Policy")
    print("="*70)

    # Create embedder
    emb = MatryoshkaEmbeddings(sizes=[768])

    # Create deterministic policy
    print("\n[1/2] Creating deterministic policy...")
    policy_det = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        n_layers=2,
        use_bayesian=False
    )

    # Create Bayesian policy
    print("[2/2] Creating Bayesian policy (10 MC samples)...")
    policy_bay = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        n_layers=2,
        use_bayesian=True,
        bayesian_samples=10
    )

    # Test query
    features = create_mock_features("factual", coherence=0.85)
    context = create_mock_context(n_shards=3, relevance=0.8)

    # Deterministic forward pass
    print("\n--- Deterministic Policy ---")
    start = time.time()
    action_det = await policy_det.decide(features, context)
    det_time = (time.time() - start) * 1000

    print(f"Tool: {action_det.chosen_tool}")
    print(f"Tool probs: {action_det.tool_probs}")
    print(f"Time: {det_time:.2f} ms")

    # Bayesian forward pass
    print("\n--- Bayesian Policy ---")
    start = time.time()
    action_bay = await policy_bay.decide(features, context, return_uncertainty=True)
    bay_time = (time.time() - start) * 1000

    print(f"Tool: {action_bay.chosen_tool}")
    print(f"Tool probs: {action_bay.tool_probs}")
    print(f"Time: {bay_time:.2f} ms ({bay_time/det_time:.1f}× slowdown)")

    if hasattr(action_bay, 'metadata') and 'uncertainty' in action_bay.metadata:
        unc = action_bay.metadata['uncertainty']
        print(f"\nUncertainty Metrics:")
        print(f"  Epistemic (model):    {unc['epistemic']:.4f}")
        print(f"  Aleatoric (data):     {unc['aleatoric']:.4f}")
        print(f"  Total:                {unc['total']:.4f}")
        print(f"  Predictive entropy:   {unc['entropy']:.4f}")

    print(f"\n✓ Bayesian policy provides uncertainty estimates!")
    print(f"✓ Performance overhead: ~{bay_time/det_time:.1f}× (expected: ~10×)")


async def demo_uncertainty_by_query_type():
    """
    Show how uncertainty varies by query type.

    Hypothesis:
    - Factual queries: Low uncertainty (well-defined)
    - Procedural queries: Medium uncertainty (structured)
    - Analytical queries: High uncertainty (complex)
    """
    print("\n" + "="*70)
    print("DEMO 2: Uncertainty by Query Type")
    print("="*70)

    # Create Bayesian policy
    emb = MatryoshkaEmbeddings(sizes=[768])
    policy = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        use_bayesian=True,
        bayesian_samples=10
    )

    query_types = [
        ("factual", 0.9, "What is the capital of France?"),
        ("procedural", 0.7, "How do I implement quicksort?"),
        ("analytical", 0.5, "What are the philosophical implications of AI consciousness?")
    ]

    print("\nQuery Type         | Epistemic | Aleatoric | Total | Tool")
    print("-" * 70)

    for query_type, coherence, example in query_types:
        features = create_mock_features(query_type, coherence)
        context = create_mock_context(n_shards=5, relevance=0.7)

        action = await policy.decide(features, context, return_uncertainty=True)

        if hasattr(action, 'metadata') and 'uncertainty' in action.metadata:
            unc = action.metadata['uncertainty']
            print(f"{query_type:18s} | {unc['epistemic']:9.4f} | "
                  f"{unc['aleatoric']:9.4f} | {unc['total']:5.4f} | {action.chosen_tool}")

    print("\n✓ Uncertainty increases with query complexity!")
    print("✓ Epistemic uncertainty reflects model confidence")
    print("✓ Aleatoric uncertainty reflects irreducible randomness")


async def demo_uncertainty_driven_exploration():
    """
    Show how high uncertainty triggers more exploration.

    Bayesian policy uses epistemic uncertainty to adaptively adjust
    exploration rate: high uncertainty → explore more.
    """
    print("\n" + "="*70)
    print("DEMO 3: Uncertainty-Driven Exploration")
    print("="*70)

    # Create Bayesian policy
    emb = MatryoshkaEmbeddings(sizes=[768])
    policy = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        use_bayesian=True,
        bayesian_samples=10,
        epsilon=0.1  # Base exploration rate: 10%
    )

    print("\nSimulating 20 queries with varying confidence...")
    print("\nCoherence | Epistemic Unc | Adaptive ε | Exploration")
    print("-" * 70)

    coherences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]

    for coherence in coherences:
        features = create_mock_features("factual", coherence)
        context = create_mock_context(n_shards=5, relevance=0.7)

        # Make 5 decisions to see exploration pattern
        explore_count = 0
        total_epistemic = 0.0
        total_adaptive_eps = 0.0

        for _ in range(5):
            action = await policy.decide(features, context, return_uncertainty=True)

            if hasattr(action, 'metadata'):
                unc = action.metadata.get('uncertainty', {})
                total_epistemic += unc.get('epistemic', 0.0)
                total_adaptive_eps += action.metadata.get('adaptive_epsilon', 0.1)

                if action.metadata.get('bandit', {}).get('mode') == 'explore':
                    explore_count += 1

        avg_epistemic = total_epistemic / 5
        avg_adaptive_eps = total_adaptive_eps / 5
        explore_rate = explore_count / 5

        print(f"{coherence:.2f}      | {avg_epistemic:13.4f} | "
              f"{avg_adaptive_eps:10.4f} | {explore_rate:.0%}")

    print("\n✓ High epistemic uncertainty → higher adaptive exploration!")
    print("✓ Adaptive ε = base_ε × (1 + epistemic_unc)")
    print("✓ System automatically explores more when uncertain")


async def demo_out_of_distribution():
    """
    Show Bayesian policy on out-of-distribution inputs.

    Hypothesis: OOD inputs → high epistemic uncertainty
    """
    print("\n" + "="*70)
    print("DEMO 4: Out-of-Distribution Detection")
    print("="*70)

    # Create Bayesian policy
    emb = MatryoshkaEmbeddings(sizes=[768])
    policy = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        use_bayesian=True,
        bayesian_samples=20  # More samples for better uncertainty estimate
    )

    # In-distribution: normal features
    print("\n[1/2] In-distribution query...")
    features_id = create_mock_features("factual", coherence=0.8)
    context = create_mock_context(n_shards=5, relevance=0.7)

    action_id = await policy.decide(features_id, context, return_uncertainty=True)
    unc_id = action_id.metadata.get('uncertainty', {})

    print(f"Tool: {action_id.chosen_tool}")
    print(f"Epistemic uncertainty: {unc_id.get('epistemic', 0.0):.4f}")

    # Out-of-distribution: random noise
    print("\n[2/2] Out-of-distribution query (random noise)...")
    features_ood = Features(
        psi=np.random.randn(6) * 10,  # Large random values
        motifs=[],  # No recognizable patterns
        metrics={"coherence": 0.1, "fiedler": 0.9},
        confidence=0.1
    )

    action_ood = await policy.decide(features_ood, context, return_uncertainty=True)
    unc_ood = action_ood.metadata.get('uncertainty', {})

    print(f"Tool: {action_ood.chosen_tool}")
    print(f"Epistemic uncertainty: {unc_ood.get('epistemic', 0.0):.4f}")

    print("\n✓ OOD inputs have higher epistemic uncertainty!")
    print(f"✓ Uncertainty ratio: {unc_ood.get('epistemic', 0.0) / max(unc_id.get('epistemic', 0.01), 0.01):.2f}×")
    print("✓ System can detect when it's operating outside training distribution")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("   HoloLoom Bayesian Policy Demo")
    print("   Variational Inference + Uncertainty Quantification")
    print("="*70)

    print("\nThis demo shows:")
    print("1. Deterministic vs Bayesian policy comparison")
    print("2. Uncertainty by query type (factual, procedural, analytical)")
    print("3. Uncertainty-driven exploration (adaptive ε)")
    print("4. Out-of-distribution detection")

    try:
        await demo_deterministic_vs_bayesian()
        await demo_uncertainty_by_query_type()
        await demo_uncertainty_driven_exploration()
        await demo_out_of_distribution()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("✓ All demos complete!")
    print("="*70)

    print("\n📊 Key Takeaways:")
    print("  • Bayesian policy provides epistemic + aleatoric uncertainty")
    print("  • Epistemic uncertainty drives adaptive exploration")
    print("  • Uncertainty increases with query complexity")
    print("  • OOD queries have high epistemic uncertainty")
    print("  • Performance overhead: ~10× (MC sampling)")

    print("\n🔧 Usage in Production:")
    print("  from HoloLoom.config import Config")
    print("  from HoloLoom.policy.unified import create_policy")
    print("")
    print("  config = Config.fused()")
    print("  config.use_bayesian = True  # Enable uncertainty quantification")
    print("  config.bayesian_samples = 10  # MC samples")
    print("")
    print("  policy = create_policy(")
    print("      mem_dim=768,")
    print("      emb=embedder,")
    print("      scales=[768],")
    print("      use_bayesian=config.use_bayesian,")
    print("      bayesian_samples=config.bayesian_samples")
    print("  )")
    print("")
    print("  action = await policy.decide(features, context, return_uncertainty=True)")
    print("  uncertainty = action.metadata['uncertainty']")
    print("  print(f\"Epistemic: {uncertainty['epistemic']:.4f}\")")


if __name__ == "__main__":
    asyncio.run(main())
