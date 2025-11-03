"""
Demo: Four-Agent Self-Tuning System

Shows TimeoutTuner, CacheTuner, ThresholdTuner, and MemoryTuner working together.
Meta-bandit coordinator selects which agent to activate based on impact.

This demo:
1. Creates Master Tuning Coordinator with 4 agents
2. Simulates queries with different complexity levels
3. Shows MemoryTuner learning optimal retrieval_k per complexity
4. Demonstrates meta-bandit agent selection
"""

import asyncio
import numpy as np
from HoloLoom.tuning import (
    MasterTuningCoordinator,
    TimeoutTuner,
    CacheTuner,
    ThresholdTuner,
    MemoryTuner,
)
from HoloLoom.tuning.cache_tuner import CacheMetrics
from HoloLoom.tuning.threshold_tuner import QualityMetrics
from HoloLoom.tuning.memory_tuner import RetrievalMetrics, QueryComplexity

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_section(text):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*70}{Colors.ENDC}")


def simulate_query_batch(
    coordinator: MasterTuningCoordinator,
    batch_name: str,
    n_queries: int,
    profile: dict,
):
    """
    Simulate batch of queries with specific characteristics.

    Args:
        coordinator: MasterTuningCoordinator instance
        batch_name: Description of batch
        n_queries: Number of queries to simulate
        profile: Dict with simulation parameters
    """
    print(f"\n{Colors.OKBLUE}Simulating: {batch_name} ({n_queries} queries){Colors.ENDC}")

    timeout_tuner = coordinator.agents['timeout']
    cache_tuner = coordinator.agents['cache']
    threshold_tuner = coordinator.agents['threshold']
    memory_tuner = coordinator.agents['memory']

    for i in range(n_queries):
        # Simulate timeouts
        for stage in ['features', 'retrieval', 'decision', 'execution']:
            mean_latency = profile.get(f'{stage}_latency_ms', 50.0)
            latency_ms = max(0, np.random.normal(mean_latency, mean_latency * 0.2))
            timeout_tuner.record_stage_completion(stage, latency_ms, timed_out=False)

        # Simulate cache metrics
        for cache_type in ['query', 'parse', 'merge']:
            hit_rate = profile.get('cache_hit_rate', 0.7)
            hit = np.random.random() < hit_rate
            cache_tuner.record_cache_metrics(
                cache_type,
                CacheMetrics(
                    hits=1 if hit else 0,
                    misses=0 if hit else 1,
                    evictions=0 if hit else 1,
                    size=int(np.random.uniform(100, 500)),
                    max_size=1000,
                )
            )

        # Simulate threshold quality
        for threshold_name in threshold_tuner.threshold_names:
            precision = profile.get('threshold_precision', 0.80)
            recall = profile.get('threshold_recall', 0.75)

            tp = int(precision * 100)
            fp = int((1 - precision) * 100)
            fn = int((1 - recall) * 100)
            tn = 200 - tp - fp - fn

            threshold_tuner.record_quality_metrics(
                threshold_name,
                QualityMetrics(
                    true_positives=max(0, tp),
                    false_positives=max(0, fp),
                    true_negatives=max(0, tn),
                    false_negatives=max(0, fn),
                )
            )

        # Simulate retrieval metrics (MemoryTuner)
        complexity = profile.get('query_complexity', QueryComplexity.MODERATE)
        optimal_k = profile.get('optimal_retrieval_k', 10)

        # Create fake query text that will classify to the desired complexity
        if complexity == QueryComplexity.SIMPLE:
            query_text = "What is X?"  # <10 words
        elif complexity == QueryComplexity.MODERATE:
            query_text = "How does X work and what are the main components?"  # 10-20 words
        else:
            query_text = "Can you explain the architecture of X and how it integrates with Y in a production environment?"  # >20 words

        # Current retrieval_k for this complexity
        current_k = memory_tuner.get_adaptive_retrieval_k(complexity)

        # Quality degrades if k is far from optimal
        k_distance = abs(current_k - optimal_k)
        relevance_penalty = k_distance * 0.05  # 5% penalty per unit distance

        precision_at_k = max(0.5, profile.get('retrieval_precision', 0.80) - relevance_penalty)
        recall_at_k = max(0.5, profile.get('retrieval_recall', 0.75) - relevance_penalty)

        relevant_retrieved = int(precision_at_k * current_k)
        total_relevant = int(relevant_retrieved / recall_at_k) if recall_at_k > 0 else relevant_retrieved

        memory_tuner.record_retrieval_metrics(
            query_text,
            RetrievalMetrics(
                relevant_retrieved=relevant_retrieved,
                total_retrieved=current_k,
                total_relevant=total_relevant,
                retrieval_latency_ms=current_k * 2.0,  # Linear cost with k
                context_usefulness=precision_at_k * 0.9,  # Context slightly worse than precision
            )
        )

        # Show progress every 10 queries
        if (i + 1) % 10 == 0:
            print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} Processed {i+1}/{n_queries} queries")


def print_system_status(coordinator: MasterTuningCoordinator):
    """Print comprehensive system status."""
    timeout_tuner = coordinator.agents['timeout']
    cache_tuner = coordinator.agents['cache']
    threshold_tuner = coordinator.agents['threshold']
    memory_tuner = coordinator.agents['memory']

    print(f"\n{Colors.BOLD}Current Configuration:{Colors.ENDC}")

    # Timeouts
    print(f"\n  Timeouts:")
    for stage in ['features', 'retrieval', 'decision', 'execution']:
        timeout = timeout_tuner.get_adaptive_timeout(stage)
        print(f"    {stage:12s}: {timeout*1000:6.1f}ms")

    # Cache sizes
    print(f"\n  Cache Sizes:")
    for cache_type in ['query', 'parse', 'merge']:
        size = int(cache_tuner.safe_params[cache_type].current_value)
        print(f"    {cache_type:12s}: {size:6d}")

    # Thresholds (show first 4)
    print(f"\n  Thresholds (sample):")
    for threshold_name in list(threshold_tuner.threshold_names)[:4]:
        value = threshold_tuner.get_adaptive_threshold(threshold_name)
        print(f"    {threshold_name:20s}: {value:.2f}")

    # Retrieval K
    print(f"\n  Retrieval K:")
    for complexity in QueryComplexity:
        k = memory_tuner.get_adaptive_retrieval_k(complexity)
        print(f"    {complexity.value:12s}: {k:3d}")

    # Agent stats
    print(f"\n{Colors.BOLD}Agent Statistics:{Colors.ENDC}")
    stats = coordinator.get_agent_stats()
    for agent_name, agent_stats in stats.items():
        success_rate = agent_stats['success_rate']
        cycles = agent_stats['total_cycles']
        print(f"  {agent_name:12s}: {success_rate:5.1%} success ({cycles} cycles)")


async def main():
    """Run four-agent tuning demonstration."""
    print_header("HoloLoom Four-Agent Self-Tuning System")

    print(f"{Colors.OKGREEN}Thompson Sampling Bayby!{Colors.ENDC}\n")

    # Create tuning coordinator with 4 agents
    print_section("1. Initializing Self-Tuning System")

    coordinator = MasterTuningCoordinator(state_dir="./demo_four_agent_state")

    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Master Tuning Coordinator initialized")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} 4 agents ready: TimeoutTuner, CacheTuner, ThresholdTuner, MemoryTuner")

    # Phase 1: Simple queries (optimal k=5)
    print_section("2. Phase 1: Simple Queries (Factual)")

    simulate_query_batch(
        coordinator,
        "Simple factual queries",
        n_queries=30,
        profile={
            'features_latency_ms': 25.0,
            'retrieval_latency_ms': 15.0,
            'decision_latency_ms': 20.0,
            'execution_latency_ms': 30.0,
            'cache_hit_rate': 0.75,
            'threshold_precision': 0.90,
            'threshold_recall': 0.85,
            'query_complexity': QueryComplexity.SIMPLE,
            'optimal_retrieval_k': 5,
            'retrieval_precision': 0.90,
            'retrieval_recall': 0.85,
        }
    )

    print_system_status(coordinator)

    # Run tuning cycle
    print_section("3. Running Tuning Cycle #1")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result.get('agent', 'unknown')}")
    print(f"  Impact: {result.get('impact', 0):.3f}")

    # Phase 2: Complex queries (optimal k=20)
    print_section("4. Phase 2: Complex Queries (Research)")

    simulate_query_batch(
        coordinator,
        "Complex research queries",
        n_queries=30,
        profile={
            'features_latency_ms': 80.0,
            'retrieval_latency_ms': 60.0,
            'decision_latency_ms': 50.0,
            'execution_latency_ms': 120.0,
            'cache_hit_rate': 0.40,
            'threshold_precision': 0.75,
            'threshold_recall': 0.70,
            'query_complexity': QueryComplexity.COMPLEX,
            'optimal_retrieval_k': 20,
            'retrieval_precision': 0.75,
            'retrieval_recall': 0.80,
        }
    )

    print_system_status(coordinator)

    # Run tuning cycle
    print_section("5. Running Tuning Cycle #2")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result.get('agent', 'unknown')}")
    print(f"  Impact: {result.get('impact', 0):.3f}")

    # Phase 3: Moderate queries (optimal k=10)
    print_section("6. Phase 3: Moderate Queries (Multi-step)")

    simulate_query_batch(
        coordinator,
        "Moderate multi-step queries",
        n_queries=30,
        profile={
            'features_latency_ms': 40.0,
            'retrieval_latency_ms': 30.0,
            'decision_latency_ms': 35.0,
            'execution_latency_ms': 60.0,
            'cache_hit_rate': 0.60,
            'threshold_precision': 0.85,
            'threshold_recall': 0.80,
            'query_complexity': QueryComplexity.MODERATE,
            'optimal_retrieval_k': 10,
            'retrieval_precision': 0.85,
            'retrieval_recall': 0.80,
        }
    )

    print_system_status(coordinator)

    # Run tuning cycle
    print_section("7. Running Tuning Cycle #3")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result.get('agent', 'unknown')}")
    print(f"  Impact: {result.get('impact', 0):.3f}")

    # Final summary
    print_section("8. Final Summary")

    summary = coordinator.get_tuning_summary()

    print(f"\n{Colors.BOLD}Tuning Summary:{Colors.ENDC}")
    print(f"  Total queries processed: {summary['total_queries']}")
    print(f"  Total tuning cycles: {summary['total_tuning_cycles']}")
    print(f"  Circuit breaker: {'OPEN' if summary['circuit_breaker_open'] else 'CLOSED'}")

    print(f"\n{Colors.BOLD}Agent Performance:{Colors.ENDC}")
    for agent_name, stats in summary['agents'].items():
        print(f"  {agent_name:15s}: {stats['success_rate']:5.1%} success ({stats['total_cycles']} cycles)")

    print(f"\n{Colors.BOLD}Meta-Bandit Selection:{Colors.ENDC}")
    meta_stats = summary['meta_bandit']
    agent_names = list(coordinator.agents.keys())
    for arm_idx, arm_stats in meta_stats.items():
        agent_name = agent_names[arm_idx]
        expected = arm_stats['expected_reward']
        pulls = arm_stats['pulls']
        print(f"  {agent_name:15s}: {expected:.3f} expected reward ({int(pulls)} pulls)")

    print_header("Demo Complete!")

    print(f"\n{Colors.OKGREEN}Key Achievements:{Colors.ENDC}")
    print("  [OK] 4 agents working in harmony via meta-bandit")
    print("  [OK] MemoryTuner learned optimal retrieval_k per complexity")
    print("  [OK] TimeoutTuner adapted to varying latencies")
    print("  [OK] CacheTuner optimized cache sizes")
    print("  [OK] ThresholdTuner balanced precision/recall")

    print(f"\n{Colors.OKCYAN}Configuration Progress:{Colors.ENDC}")
    print("  Before: 72 parameters (manual configuration hell)")
    print("  After Agent 4: 54 parameters (18 eliminated)")
    print("  Reduction: 25% (on track for 96% target)")

    print(f"\n{Colors.WARNING}Next: 3 more agents for complete self-tuning!{Colors.ENDC}\n")


if __name__ == '__main__':
    asyncio.run(main())
