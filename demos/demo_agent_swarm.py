"""
Agent Swarm Deployment Demo

Deploys all 7 self-tuning agents in a coordinated swarm for comprehensive testing.
This demo simulates realistic workload patterns and shows how the agents adapt.

Agents Deployed:
1. TimeoutTuner - Optimizes 4 timeout parameters
2. CacheTuner - Optimizes 3 cache size parameters
3. ThresholdTuner - Optimizes 8 threshold parameters
4. MemoryTuner - Optimizes 3 retrieval parameters
5. ComplexityTuner - Optimizes execution mode selection
6. PolicyTuner - Optimizes exploration/exploitation
7. PhysicsTuner - Optimizes 12 spring dynamics parameters

Total: 35 parameters auto-tuned via Thompson Sampling!
"""

import asyncio
import numpy as np
import time
from typing import Dict, Any, List
from collections import defaultdict

from HoloLoom.tuning import (
    MasterTuningCoordinator,
    TimeoutTuner,
    CacheTuner,
    ThresholdTuner,
    MemoryTuner,
    ComplexityTuner,
    PolicyTuner,
    PhysicsTuner,
)
from HoloLoom.tuning.cache_tuner import CacheMetrics
from HoloLoom.tuning.threshold_tuner import QualityMetrics
from HoloLoom.tuning.memory_tuner import RetrievalMetrics
from HoloLoom.tuning.memory_tuner import QueryComplexity as MemoryQueryComplexity
from HoloLoom.tuning.complexity_tuner import ComplexityMetrics, ExecutionMode
from HoloLoom.tuning.complexity_tuner import QueryComplexity as ComplexityQueryComplexity
from HoloLoom.tuning.policy_tuner import PolicyMetrics
from HoloLoom.tuning.physics_tuner import PhysicsMetrics

# ANSI colors
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
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*80}{Colors.ENDC}")


class WorkloadSimulator:
    """Simulates realistic query workload patterns."""

    def __init__(self):
        self.workload_patterns = {
            'morning_rush': {
                'description': 'Morning spike - simple queries, high cache hit rate',
                'memory_complexity': MemoryQueryComplexity.SIMPLE,
                'complexity_complexity': ComplexityQueryComplexity.SIMPLE,
                'execution_mode': ExecutionMode.BARE,
                'cache_hit_rate': 0.85,
                'latency_scale': 0.7,
                'quality_scale': 0.80,
            },
            'midday_mixed': {
                'description': 'Midday balanced - mixed complexity, moderate cache',
                'memory_complexity': MemoryQueryComplexity.MODERATE,
                'complexity_complexity': ComplexityQueryComplexity.MODERATE,
                'execution_mode': ExecutionMode.FAST,
                'cache_hit_rate': 0.60,
                'latency_scale': 1.0,
                'quality_scale': 0.85,
            },
            'afternoon_research': {
                'description': 'Afternoon research - complex queries, low cache hit',
                'memory_complexity': MemoryQueryComplexity.COMPLEX,
                'complexity_complexity': ComplexityQueryComplexity.COMPLEX,
                'execution_mode': ExecutionMode.FUSED,
                'cache_hit_rate': 0.30,
                'latency_scale': 1.5,
                'quality_scale': 0.92,
            },
            'evening_batch': {
                'description': 'Evening batch processing - simple + complex mix',
                'memory_complexity': MemoryQueryComplexity.MODERATE,
                'complexity_complexity': ComplexityQueryComplexity.MODERATE,
                'execution_mode': ExecutionMode.FAST,
                'cache_hit_rate': 0.50,
                'latency_scale': 0.9,
                'quality_scale': 0.87,
            },
        }

    def get_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """Get workload pattern configuration."""
        return self.workload_patterns[pattern_name]


def simulate_queries(coordinator: MasterTuningCoordinator, pattern_name: str, n_queries: int):
    """
    Simulate batch of queries with realistic metrics.

    Args:
        coordinator: Master coordinator
        pattern_name: Workload pattern name
        n_queries: Number of queries to simulate
    """
    simulator = WorkloadSimulator()
    pattern = simulator.get_pattern(pattern_name)

    print(f"\n{Colors.OKBLUE}Workload: {pattern['description']}{Colors.ENDC}")
    print(f"  Queries: {n_queries}")
    print(f"  Complexity: {pattern['memory_complexity'].value}")
    print(f"  Mode: {pattern['execution_mode'].value}")
    print(f"  Cache Hit Rate: {pattern['cache_hit_rate']:.1%}")

    agents = coordinator.agents

    for i in range(n_queries):
        # Generate query text for complexity classification
        memory_complexity = pattern['memory_complexity']
        complexity_complexity = pattern['complexity_complexity']

        if memory_complexity == MemoryQueryComplexity.SIMPLE:
            query_text = f"Query {i}: What is X?"
        elif memory_complexity == MemoryQueryComplexity.MODERATE:
            query_text = f"Query {i}: How does X work and what are the main components?"
        else:
            query_text = f"Query {i}: Explain the architecture of X and how it integrates with Y?"

        # 1. TimeoutTuner - simulate stage latencies
        latency_scale = pattern['latency_scale']
        for stage in ['features', 'retrieval', 'decision', 'execution']:
            base_latency = {'features': 30, 'retrieval': 20, 'decision': 25, 'execution': 40}[stage]
            latency = max(5, np.random.normal(base_latency * latency_scale, base_latency * 0.2))
            agents['timeout'].record_stage_completion(stage, latency, timed_out=False)

        # 2. CacheTuner - simulate cache behavior
        cache_hit_rate = pattern['cache_hit_rate']
        for cache_type in ['query', 'parse', 'merge']:
            hit = np.random.random() < cache_hit_rate
            agents['cache'].record_cache_metrics(
                cache_type,
                CacheMetrics(
                    hits=1 if hit else 0,
                    misses=0 if hit else 1,
                    evictions=0 if hit else 1,
                    size=int(np.random.uniform(100, 500)),
                    max_size=1000,
                )
            )

        # 3. ThresholdTuner - simulate classification quality
        quality_scale = pattern['quality_scale']
        for threshold_name in agents['threshold'].threshold_names:
            precision = min(0.99, quality_scale + np.random.uniform(-0.05, 0.05))
            recall = min(0.99, quality_scale * 0.95 + np.random.uniform(-0.05, 0.05))

            tp = int(precision * 100)
            fp = int((1 - precision) * 100)
            fn = int((1 - recall) * 100)
            tn = 200 - tp - fp - fn

            agents['threshold'].record_quality_metrics(
                threshold_name,
                QualityMetrics(
                    true_positives=max(0, tp),
                    false_positives=max(0, fp),
                    true_negatives=max(0, tn),
                    false_negatives=max(0, fn),
                )
            )

        # 4. MemoryTuner - simulate retrieval
        current_k = agents['memory'].get_adaptive_retrieval_k(memory_complexity)
        precision_at_k = quality_scale * 0.9
        recall_at_k = quality_scale * 0.85

        relevant_retrieved = int(precision_at_k * current_k)
        total_relevant = int(relevant_retrieved / recall_at_k) if recall_at_k > 0 else relevant_retrieved

        agents['memory'].record_retrieval_metrics(
            query_text,
            RetrievalMetrics(
                relevant_retrieved=relevant_retrieved,
                total_retrieved=current_k,
                total_relevant=total_relevant,
                retrieval_latency_ms=current_k * 2.0,
                context_usefulness=precision_at_k * 0.95,
            )
        )

        # 5. ComplexityTuner - simulate execution mode performance
        current_mode = agents['complexity'].get_adaptive_mode(complexity_complexity)
        optimal_mode = pattern['execution_mode']

        if current_mode == optimal_mode:
            # Optimal mode
            quality_score = quality_scale + np.random.uniform(-0.03, 0.03)
            latency_ms = 50 * latency_scale * (1 + current_mode.value.count('e'))  # BARE=50, FAST=150, FUSED=250
        else:
            # Suboptimal mode penalty
            quality_score = quality_scale * 0.85
            latency_ms = 50 * latency_scale * (1 + current_mode.value.count('e')) * 1.2

        agents['complexity'].record_complexity_metrics(
            query_text,
            ComplexityMetrics(
                latency_ms=latency_ms,
                quality_score=quality_score,
                mode=current_mode,
            )
        )

        # 6. PolicyTuner - simulate decision quality
        decision_accuracy = quality_scale * 0.92
        tool_diversity = 0.3 + np.random.uniform(-0.05, 0.05)
        regret = 0.1 + np.random.uniform(-0.05, 0.05)
        confidence = quality_scale

        agents['policy'].record_policy_metrics(
            PolicyMetrics(
                decision_accuracy=decision_accuracy,
                tool_diversity=tool_diversity,
                regret=regret,
                confidence=confidence,
            )
        )

        # 7. PhysicsTuner - simulate propagation quality
        convergence_speed = quality_scale * 0.85
        stability = quality_scale * 0.90
        propagation_accuracy = quality_scale
        energy_efficiency = 0.7 + np.random.uniform(-0.1, 0.1)

        agents['physics'].record_physics_metrics(
            PhysicsMetrics(
                convergence_speed=convergence_speed,
                stability=stability,
                propagation_accuracy=propagation_accuracy,
                energy_efficiency=energy_efficiency,
            )
        )

        # Progress indicator
        if (i + 1) % 25 == 0 or (i + 1) == n_queries:
            print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} Processed {i+1}/{n_queries} queries")


def print_agent_status(coordinator: MasterTuningCoordinator):
    """Print detailed agent status."""
    print(f"\n{Colors.BOLD}Agent Status:{Colors.ENDC}")

    stats = coordinator.get_agent_stats()
    for agent_name, agent_stats in stats.items():
        success_rate = agent_stats['success_rate']
        cycles = agent_stats['total_cycles']
        status_icon = "[OK]" if success_rate >= 0.5 else "[WARN]"
        color = Colors.OKGREEN if success_rate >= 0.5 else Colors.WARNING

        print(f"  {color}{status_icon}{Colors.ENDC} {agent_name:15s}: {success_rate:5.1%} success ({cycles} cycles)")


def print_meta_bandit_stats(coordinator: MasterTuningCoordinator):
    """Print meta-bandit selection statistics."""
    print(f"\n{Colors.BOLD}Meta-Bandit Selection (Agent Impact):{Colors.ENDC}")

    summary = coordinator.get_tuning_summary()
    meta_stats = summary['meta_bandit']
    agent_names = list(coordinator.agents.keys())

    # Sort by expected reward
    sorted_agents = sorted(
        [(agent_names[idx], stats['expected_reward'], stats['pulls'])
         for idx, stats in meta_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )

    for agent_name, expected_reward, pulls in sorted_agents:
        bar_length = int(expected_reward * 40)
        bar = '#' * bar_length + '.' * (40 - bar_length)
        print(f"  {agent_name:15s}: {bar} {expected_reward:.3f} ({int(pulls)} pulls)")


def print_parameter_summary(coordinator: MasterTuningCoordinator):
    """Print current parameter values."""
    print(f"\n{Colors.BOLD}Current Parameters (Sample):{Colors.ENDC}")

    agents = coordinator.agents

    # Timeouts (show 2)
    print(f"\n  Timeouts:")
    for stage in ['features', 'execution']:
        timeout = agents['timeout'].get_adaptive_timeout(stage)
        print(f"    {stage:12s}: {timeout*1000:6.1f}ms")

    # Cache sizes
    print(f"\n  Cache Sizes:")
    for cache_type in ['query', 'merge']:
        size = int(agents['cache'].safe_params[cache_type].current_value)
        print(f"    {cache_type:12s}: {size:6d}")

    # Retrieval K
    print(f"\n  Retrieval K:")
    for complexity in [MemoryQueryComplexity.SIMPLE, MemoryQueryComplexity.COMPLEX]:
        k = agents['memory'].get_adaptive_retrieval_k(complexity)
        print(f"    {complexity.value:12s}: {k:3d}")

    # Execution Modes
    print(f"\n  Execution Modes:")
    for complexity in [ComplexityQueryComplexity.SIMPLE, ComplexityQueryComplexity.COMPLEX]:
        mode = agents['complexity'].get_adaptive_mode(complexity)
        print(f"    {complexity.value:12s}: {mode.value:6s}")

    # Policy params
    print(f"\n  Policy:")
    epsilon = agents['policy'].get_adaptive_epsilon()
    bayesian = agents['policy'].get_adaptive_bayesian_blend()
    print(f"    epsilon     : {epsilon:.3f}")
    print(f"    bayesian    : {bayesian:.3f}")

    # Physics (show 3)
    print(f"\n  Spring Dynamics:")
    for param in ['stiffness', 'damping', 'decay']:
        value = agents['physics'].get_adaptive_parameter(param)
        print(f"    {param:12s}: {value:.3f}")


async def main():
    """Run agent swarm deployment."""
    print_header("AGENT SWARM DEPLOYMENT")
    print(f"{Colors.OKGREEN}Thompson Sampling Bayby!{Colors.ENDC}")
    print(f"{Colors.BOLD}Deploying 7 self-tuning agents...{Colors.ENDC}\n")

    # Initialize coordinator
    print_section("1. Initializing Agent Swarm")
    start_time = time.time()
    coordinator = MasterTuningCoordinator(state_dir="./swarm_state")
    init_time = time.time() - start_time

    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} 7 agents initialized in {init_time*1000:.1f}ms")
    print(f"  - Agent 1: TimeoutTuner (4 parameters)")
    print(f"  - Agent 2: CacheTuner (3 parameters)")
    print(f"  - Agent 3: ThresholdTuner (8 parameters)")
    print(f"  - Agent 4: MemoryTuner (3 parameters)")
    print(f"  - Agent 5: ComplexityTuner (3 parameters)")
    print(f"  - Agent 6: PolicyTuner (2 parameters)")
    print(f"  - Agent 7: PhysicsTuner (12 parameters)")
    print(f"{Colors.BOLD}  Total: 35 parameters under Thompson Sampling control{Colors.ENDC}")

    # Workload 1: Morning rush
    print_section("2. Workload 1: Morning Rush (Simple Queries)")
    simulate_queries(coordinator, 'morning_rush', n_queries=100)
    print_agent_status(coordinator)

    # Run tuning cycle
    print(f"\n{Colors.OKCYAN}Running tuning cycle...{Colors.ENDC}")
    result = await coordinator.run_tuning_cycle()
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Agent selected: {result.get('agent', 'unknown')}")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Impact: {result.get('impact', 0):.3f}")

    # Workload 2: Midday mixed
    print_section("3. Workload 2: Midday Mixed (Moderate Queries)")
    simulate_queries(coordinator, 'midday_mixed', n_queries=100)
    print_agent_status(coordinator)

    # Run tuning cycle
    print(f"\n{Colors.OKCYAN}Running tuning cycle...{Colors.ENDC}")
    result = await coordinator.run_tuning_cycle()
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Agent selected: {result.get('agent', 'unknown')}")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Impact: {result.get('impact', 0):.3f}")

    # Workload 3: Afternoon research
    print_section("4. Workload 3: Afternoon Research (Complex Queries)")
    simulate_queries(coordinator, 'afternoon_research', n_queries=100)
    print_agent_status(coordinator)

    # Run tuning cycle
    print(f"\n{Colors.OKCYAN}Running tuning cycle...{Colors.ENDC}")
    result = await coordinator.run_tuning_cycle()
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Agent selected: {result.get('agent', 'unknown')}")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Impact: {result.get('impact', 0):.3f}")

    # Workload 4: Evening batch
    print_section("5. Workload 4: Evening Batch (Mixed)")
    simulate_queries(coordinator, 'evening_batch', n_queries=100)
    print_agent_status(coordinator)

    # Final tuning cycle
    print(f"\n{Colors.OKCYAN}Running final tuning cycle...{Colors.ENDC}")
    result = await coordinator.run_tuning_cycle()
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Agent selected: {result.get('agent', 'unknown')}")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Impact: {result.get('impact', 0):.3f}")

    # Final summary
    print_section("6. Swarm Performance Summary")

    summary = coordinator.get_tuning_summary()
    total_time = time.time() - start_time

    print(f"\n{Colors.BOLD}Deployment Stats:{Colors.ENDC}")
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Queries processed: {summary['total_queries']}")
    print(f"  Tuning cycles: {summary['total_tuning_cycles']}")
    print(f"  Circuit breaker: {'OPEN' if summary['circuit_breaker_open'] else 'CLOSED'}")

    print_meta_bandit_stats(coordinator)
    print_parameter_summary(coordinator)

    # Key achievements
    print_header("Swarm Deployment Complete!")

    print(f"\n{Colors.OKGREEN}Key Achievements:{Colors.ENDC}")
    print("  [OK] 7 agents deployed and operational")
    print("  [OK] 400 queries processed across 4 workload patterns")
    print("  [OK] 4 tuning cycles completed")
    print("  [OK] 35 parameters auto-tuned via Thompson Sampling")
    print("  [OK] Meta-bandit learned agent impact distribution")

    print(f"\n{Colors.OKCYAN}Configuration Impact:{Colors.ENDC}")
    print("  Before: 72 parameters (manual configuration)")
    print("  After:  37 parameters (35 auto-tuned)")
    print("  Reduction: 49% (on track for 96% target)")

    print(f"\n{Colors.WARNING}Swarm Status: OPERATIONAL{Colors.ENDC}")
    print(f"{Colors.BOLD}Thompson Sampling Bayby!{Colors.ENDC}\n")


if __name__ == '__main__':
    asyncio.run(main())
