"""
Routing Strategies Demo - A/B Testing Learnable Backend Selection
=================================================================

Demonstrates:
1. Rule-based routing (baseline)
2. Learned routing with Thompson Sampling
3. A/B testing to determine winner
4. Performance comparison and lift analysis
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Routing system
from HoloLoom.memory.routing import (
    RuleBasedRouter,
    LearnedRouter,
    RoutingExperiment,
    BackendType,
    RoutingOutcome
)


def create_test_queries() -> List[str]:
    """Create diverse test queries."""
    return [
        # Relationship queries (should route to Neo4j)
        "who inspected the hives yesterday",
        "when was the last treatment applied",
        "where are the strongest colonies located",
        "what is connected to varroa mites",

        # Similarity queries (should route to Qdrant)
        "find similar inspection reports",
        "search for honey production data",
        "look for pest management strategies",
        "compare colony health metrics",

        # Personal queries (should route to Mem0)
        "my favorite hive management techniques",
        "what are my beekeeping preferences",
        "i need personal recommendations",
        "show me user-specific data",

        # Temporal queries (should route to InMemory)
        "recent hive inspections",
        "today's temperature readings",
        "latest queen activity",
        "current honey flow status",

        # Mixed/ambiguous queries
        "hive health status",
        "colony performance",
        "seasonal patterns",
        "best practices",
    ]


def simulate_query_outcome(query: str, backend: BackendType) -> float:
    """
    Simulate query execution and return relevance score.

    In production, this would be actual query execution with real feedback.
    For demo, we simulate based on whether backend is optimal for query type.
    """
    # Optimal backend mapping (ground truth)
    optimal_map = {
        "who": BackendType.NEO4J,
        "when": BackendType.NEO4J,
        "where": BackendType.NEO4J,
        "connected": BackendType.NEO4J,

        "find": BackendType.QDRANT,
        "similar": BackendType.QDRANT,
        "search": BackendType.QDRANT,
        "compare": BackendType.QDRANT,

        "my": BackendType.MEM0,
        "user": BackendType.MEM0,
        "personal": BackendType.MEM0,
        "preference": BackendType.MEM0,

        "recent": BackendType.INMEMORY,
        "today": BackendType.INMEMORY,
        "latest": BackendType.INMEMORY,
        "current": BackendType.INMEMORY,
    }

    # Check if backend is optimal
    query_lower = query.lower()
    optimal_backend = None

    for keyword, opt_backend in optimal_map.items():
        if keyword in query_lower:
            optimal_backend = opt_backend
            break

    # Relevance score based on match
    if optimal_backend == backend:
        return 0.9  # High relevance for optimal match
    elif backend == BackendType.QDRANT:
        return 0.7  # Qdrant is decent fallback
    else:
        return 0.5  # Suboptimal match


async def run_experiment():
    """Run complete routing strategy experiment."""
    print("\n" + "="*70)
    print("ROUTING STRATEGIES EXPERIMENT")
    print("="*70 + "\n")

    # Available backends (simulating production setup)
    available_backends = [
        BackendType.NEO4J,
        BackendType.QDRANT,
        BackendType.MEM0,
        BackendType.INMEMORY,
    ]

    print(f"Available backends: {[b.value for b in available_backends]}\n")

    # ========================================================================
    # Phase 1: Baseline (Rule-Based) Performance
    # ========================================================================

    print("="*70)
    print("PHASE 1: Baseline Rule-Based Routing")
    print("="*70 + "\n")

    rule_router = RuleBasedRouter()
    test_queries = create_test_queries()

    print(f"Testing {len(test_queries)} queries...\n")

    for query in test_queries:
        # Get routing decision
        decision = rule_router.select_backend(query, available_backends)

        # Simulate query execution
        relevance = simulate_query_outcome(query, decision.backend_type)
        latency = 1000 + (100 if decision.backend_type == BackendType.NEO4J else 0)

        # Record outcome
        outcome = RoutingOutcome(
            decision=decision,
            query=query,
            result_count=5,
            avg_relevance=relevance,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        rule_router.record_outcome(outcome)

        print(f"  {query[:50]:<50} -> {decision.backend_type.value:10} "
              f"(relevance: {relevance:.2f})")

    # Get baseline stats
    baseline_stats = rule_router.get_statistics()
    print(f"\n[BASELINE RESULTS]")
    print(f"  Accuracy: {baseline_stats['accuracy']:.1%}")
    print(f"  Backend distribution:")
    for backend, count in baseline_stats['backend_counts'].items():
        print(f"    {backend}: {count} queries")

    # ========================================================================
    # Phase 2: Learned Routing Performance
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 2: Learned Routing with Thompson Sampling")
    print("="*70 + "\n")

    learned_router = LearnedRouter(backends=available_backends)

    print(f"Training on {len(test_queries)} queries...\n")

    for query in test_queries:
        # Get routing decision
        decision = learned_router.select_backend(query, available_backends)

        # Simulate query execution
        relevance = simulate_query_outcome(query, decision.backend_type)
        latency = 1000 + (100 if decision.backend_type == BackendType.NEO4J else 0)

        # Record outcome (this updates Thompson Sampling)
        outcome = RoutingOutcome(
            decision=decision,
            query=query,
            result_count=5,
            avg_relevance=relevance,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        learned_router.record_outcome(outcome)

        print(f"  {query[:50]:<50} -> {decision.backend_type.value:10} "
              f"(relevance: {relevance:.2f}, conf: {decision.confidence:.2f})")

    # Get learned stats
    learned_stats = learned_router.get_statistics()
    print(f"\n[LEARNED RESULTS]")
    print(f"  Overall accuracy: {learned_stats.get('overall_accuracy', 0):.1%}")

    # ========================================================================
    # Phase 3: A/B Test Comparison
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 3: A/B Test - Rule-Based vs Learned")
    print("="*70 + "\n")

    # Create fresh routers for fair comparison
    rule_router_ab = RuleBasedRouter()
    learned_router_ab = LearnedRouter(backends=available_backends)

    # Setup experiment
    experiment = RoutingExperiment()
    experiment.add_baseline(rule_router_ab, name="rule_based")
    experiment.add_challenger("learned_thompson", learned_router_ab, weight=0.5)

    print(f"Running A/B test with 50/50 split...\n")

    # Run queries through experiment (50/50 split)
    for query in test_queries * 2:  # Run twice for more data
        # Route query
        decision = experiment.route(query, available_backends)

        # Simulate execution
        relevance = simulate_query_outcome(query, decision.backend_type)
        latency = 1000 + (100 if decision.backend_type == BackendType.NEO4J else 0)

        # Record outcome
        outcome = RoutingOutcome(
            decision=decision,
            query=query,
            result_count=5,
            avg_relevance=relevance,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        experiment.record(outcome)

    # Generate report
    report = experiment.generate_report()

    print("[A/B TEST RESULTS]")
    print(f"  Winner: {report['winner']}")
    print(f"\n  Performance by variant:")

    for variant_name, metrics in report['experiment_stats']['variants'].items():
        print(f"\n    {variant_name}:")
        print(f"      Success rate: {metrics['success_rate']:.1%}")
        print(f"      Avg relevance: {metrics['avg_relevance']:.2f}")
        print(f"      Avg latency: {metrics['avg_latency']:.0f}ms")
        print(f"      Total queries: {metrics['total_queries']}")

        if 'lift_over_baseline' in metrics:
            print(f"      Lift over baseline: {metrics['lift_over_baseline']:+.1f}%")

    print(f"\n  Recommendation: {report['recommendation']}")

    # ========================================================================
    # Phase 4: Save Learned Parameters
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 4: Persistence")
    print("="*70 + "\n")

    # Save learned router parameters
    save_path = "memory_data/learned_router.json"
    learned_router.save(save_path)
    print(f"  Learned parameters saved to: {save_path}")

    # Demonstrate loading
    loaded_router = LearnedRouter(backends=available_backends)
    loaded_router.load(save_path)
    print(f"  Parameters loaded successfully")

    # Verify it works
    test_query = "find similar hive inspections"
    decision = loaded_router.select_backend(test_query, available_backends)
    print(f"\n  Test query: '{test_query}'")
    print(f"  Routed to: {decision.backend_type.value}")
    print(f"  Confidence: {decision.confidence:.2f}")

    print("\n" + "="*70)
    print("[EXPERIMENT COMPLETE]")
    print("="*70 + "\n")

    print("Key Insights:")
    print("  1. Rule-based routing provides good baseline")
    print("  2. Learned routing adapts to actual performance")
    print("  3. A/B testing enables data-driven decisions")
    print("  4. Parameters can be saved/loaded for production")
    print("\nNext Steps:")
    print("  - Deploy winning strategy to production")
    print("  - Continue learning from real user feedback")
    print("  - Monitor performance and adjust as needed")


if __name__ == "__main__":
    asyncio.run(run_experiment())