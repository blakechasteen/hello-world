"""
Quick GP Bandits Integration Test
==================================
Minimal test to verify GP bandits work with HoloLoom.

Run:
    PYTHONPATH=. python demos/demo_gp_quick_test.py
"""

import asyncio
import numpy as np

from hololoom.config import Config
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.documentation.types import Features, Context
from hololoom.policy.gp_policy import create_gp_policy, GPConfig


async def main():
    print("=" * 70)
    print("GP BANDITS INTEGRATION TEST")
    print("=" * 70)

    # Create embeddings
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Create GP config
    gp_config = GPConfig(
        acquisition="thompson",
        kernel_type="matern",
        action_space_bounds={
            'stiffness': (0.05, 0.5),
            'damping': (0.5, 0.95),
            'temperature': (0.1, 2.0)
        },
        n_candidates_per_dim=3  # Small for speed
    )

    # Create config with safety disabled for testing
    config = Config.fast()
    config.enable_safety_guardrails = False

    # Create GP policy
    print("\nCreating GP policy...")
    policy = create_gp_policy(
        mem_dim=384,
        emb=emb,
        scales=[96, 192, 384],
        gp_config=gp_config,
        cfg=config
    )
    print(f"[OK] GP policy created with {gp_config.acquisition} acquisition")

    # Mock features and context
    features = Features(
        psi=np.random.randn(384).astype(np.float32),  # Match mem_dim
        motifs=["question→answer"],
        metrics={"coherence": 0.7, "fiedler": 0.3},
        confidence=0.8
    )

    context = Context(
        hits=[],
        kg_sub=None,
        shard_texts=["Test context"],
        relevance=0.7
    )

    # Run a few iterations
    print("\nRunning 5 iterations...")
    for i in range(5):
        action_plan = await policy.decide(features, context)
        hyperparams = policy.current_hyperparams

        print(f"\nIteration {i+1}:")
        print(f"  Tool: {action_plan.chosen_tool}")
        print(f"  Hyperparams: stiffness={hyperparams['stiffness']:.3f}, "
              f"damping={hyperparams['damping']:.3f}, "
              f"temperature={hyperparams['temperature']:.3f}")

    # Get GP statistics
    print("\nGP Statistics:")
    stats = policy.get_gp_statistics()
    print(f"  Observations: {stats['n_observations']}")
    print(f"  Best hyperparams: {stats['best_hyperparams']}")
    print(f"  Best mean: {stats['best_mean']:.3f}")
    print(f"  Best std: {stats['best_std']:.3f}")

    print("\n" + "=" * 70)
    print("[PASS] TEST PASSED - GP Bandits Integration Working!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
