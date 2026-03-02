"""
Demo: MCTS Shuttle v2.0 - Warp-Yarn Intersection

Demonstrates the complete Shuttle system:
1. Warp search (semantic/vector)
2. Policy selection (Thompson Sampling)
3. Yarn graph expansion (MCTS)
4. Hybrid context generation

Run:
    python demos/demo_shuttle_mcts.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from hololoom.shuttle import create_hololoom_shuttle


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


async def main():
    print_section("MCTS Shuttle v2.0 Demo")

    print("Creating Shuttle with mock backends (no Docker needed)...")
    print("  - Warp: Mock vector search")
    print("  - Yarn: Mock knowledge graph")
    print("  - MCTS: 50 simulations per query")
    print("  - Bandit: Thompson Sampling with Gamma-based Beta\n")

    # Create shuttle (uses mock backends)
    shuttle = create_hololoom_shuttle(
        num_mcts_simulations=50,
        enable_learning=True,
    )

    # Example queries
    queries = [
        "What's blocking the integration testing?",
        "Who is working on test data generation?",
        "What happened before the deployment?",
    ]

    for i, query in enumerate(queries, 1):
        print_section(f"Query {i}: {query}")

        # Execute query
        result = shuttle.intersect(query)

        print(f"[OK] Search completed in {result.search_time_ms:.1f}ms\n")

        print("[WARP] Fuzzy Evidence (Vector Search Results):")
        print("-" * 70)
        for j, evidence in enumerate(result.fuzzy_evidence, 1):
            print(f"  {j}. {evidence['text']}")
            print(f"     Score: {evidence['score']:.2f} | ID: {evidence['id']}")
        print()

        print("[YARN] Structural Claims (Knowledge Graph Context):")
        print("-" * 70)
        print(result.structural_claims)
        print()

        print("[META] Query Metadata:")
        print("-" * 70)
        print(f"  Policy Used: {result.policy_used}")
        print(f"  Nodes Selected: {len(result.selected_nodes)}")
        print(f"  MCTS Simulations: {result.metadata['mcts_simulations']}")
        print(f"  Reward: {result.reward:.2f}")
        print()

        # Show policy config
        config = result.metadata['policy_config']
        print("[CFG] Policy Configuration:")
        print(f"  Max Depth: {config['max_depth']}")
        print(f"  Max Nodes: {config['max_nodes']}")
        print(f"  Edge Types: {', '.join(config['allowed_edge_types'])}")
        print()

    # Show learning statistics
    print_section("Thompson Sampling Statistics (After 3 Queries)")

    stats = shuttle.get_policy_statistics()

    for policy_name, policy_stats in stats.items():
        total = policy_stats['total_pulls']
        if total > 0:
            print(f"  {policy_name}:")
            print(f"    Pulls: {total}")
            print(f"    Successes: {policy_stats['successes']}")
            print(f"    Failures: {policy_stats['failures']}")
            print(f"    Mean Reward: {policy_stats['mean_reward']:.2f}")
            print()

    # Show best policies
    print("[TOP] Top 3 Policies by Mean Reward:")
    print("-" * 70)

    best_policies = shuttle.get_best_policies(top_k=3)
    for rank, (policy_name, mean_reward) in enumerate(best_policies, 1):
        print(f"  {rank}. {policy_name}: {mean_reward:.2f}")

    print()

    # Save bandit state
    print("[SAVE] Saving bandit state to .shuttle_state.json...")
    shuttle.save_bandit_state(".shuttle_state.json")
    print("       Bandit state saved (can be loaded for future sessions)\n")

    print_section("Demo Complete!")
    print("Key Features Demonstrated:")
    print("  [OK] Gamma-based Beta sampling (numerically stable)")
    print("  [OK] MCTS with proper neighbor_map threading")
    print("  [OK] Thompson Sampling policy selection")
    print("  [OK] Warp + Yarn hybrid context generation")
    print("  [OK] Bandit learning from query outcomes")
    print()

    print("Next Steps:")
    print("  1. Test with real HoloLoom backends (Qdrant + Neo4j)")
    print("  2. Fine-tune MCTS simulations and exploration weight")
    print("  3. Improve rollout function with learned value estimator")
    print("  4. Add user feedback loop for bandit updates")
    print()


if __name__ == "__main__":
    asyncio.run(main())
