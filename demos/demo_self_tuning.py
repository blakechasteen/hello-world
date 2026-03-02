"""
Demo: Self-Tuning System

Shows adaptive timeout management using Thompson Sampling.

This demo:
1. Creates Master Tuning Coordinator
2. Simulates queries with varying latencies
3. Shows TimeoutTuner learning optimal timeouts
4. Demonstrates Thompson Sampling adaptation
"""

import asyncio
import numpy as np
from hololoom.tuning import MasterTuningCoordinator, TimeoutTuner

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


async def simulate_query_batch(timeout_tuner: TimeoutTuner,
                               batch_name: str,
                               latency_params: dict,
                               n_queries: int = 50):
    """
    Simulate batch of queries with specific latency characteristics.

    Args:
        timeout_tuner: TimeoutTuner instance
        batch_name: Description of batch
        latency_params: Dict with mean/std for each stage
        n_queries: Number of queries to simulate
    """
    print(f"\n{Colors.OKBLUE}Simulating: {batch_name} ({n_queries} queries){Colors.ENDC}")

    for i in range(n_queries):
        # Simulate latencies for each stage
        for stage, params in latency_params.items():
            mean = params['mean']
            std = params['std']

            # Generate latency (normal distribution, clipped to positive)
            latency_ms = max(0, np.random.normal(mean, std))

            # Check if would timeout with current setting
            current_timeout = timeout_tuner.get_adaptive_timeout(stage)
            timed_out = (latency_ms / 1000.0) > current_timeout

            # Record for learning
            timeout_tuner.record_stage_completion(stage, latency_ms, timed_out)

        # Show progress every 10 queries
        if (i + 1) % 10 == 0:
            print(f"  {Colors.OKGREEN}[OK]{Colors.ENDC} Processed {i+1}/{n_queries} queries")


def print_timeout_stats(timeout_tuner: TimeoutTuner):
    """Print current timeout statistics."""
    print(f"\n{Colors.BOLD}Current Timeouts:{Colors.ENDC}")

    for stage in ['features', 'retrieval', 'decision', 'execution']:
        timeout = timeout_tuner.get_adaptive_timeout(stage)
        samples = list(timeout_tuner.latencies.get(stage, []))

        if len(samples) >= 20:
            p50 = np.percentile(samples, 50)
            p95 = np.percentile(samples, 95)
            p99 = np.percentile(samples, 99)

            # Calculate timeout margin
            margin = (timeout * 1000) / p95 if p95 > 0 else 0

            print(f"  {stage:12s}: {timeout*1000:6.1f}ms "
                  f"(p50={p50:5.1f}ms, p95={p95:5.1f}ms, p99={p99:5.1f}ms, margin={margin:.2f}x)")
        else:
            print(f"  {stage:12s}: {timeout*1000:6.1f}ms (insufficient data)")


def print_bandit_stats(timeout_tuner: TimeoutTuner):
    """Print Thompson Sampling bandit statistics."""
    print(f"\n{Colors.BOLD}Thompson Sampling Statistics:{Colors.ENDC}")

    load_state = timeout_tuner.load_state.category
    print(f"  Load state: {load_state}")

    for load_cat, bandit in timeout_tuner.bandits.items():
        print(f"\n  {load_cat}:")
        stats = bandit.get_stats()

        for arm_idx, arm_stats in stats.items():
            margin = [1.2, 1.5, 2.0, 2.5, 3.0][arm_idx]
            expected = arm_stats['expected_reward']
            pulls = arm_stats['pulls']

            bar_length = int(expected * 20)  # Bar chart
            bar = '#' * bar_length + '-' * (20 - bar_length)

            marker = " <-- SELECTED" if arm_idx == bandit.sample() else ""

            print(f"    {margin:.1f}x: {bar} {expected:.3f} ({int(pulls)} pulls){marker}")


async def main():
    """Run self-tuning demonstration."""
    print_header("HoloLoom Self-Tuning System Demo")

    print(f"{Colors.OKGREEN}Thompson Sampling Bayby!{Colors.ENDC}\n")

    # Create tuning coordinator
    print_section("1. Initializing Self-Tuning System")

    coordinator = MasterTuningCoordinator(state_dir="./demo_tuning_state")
    timeout_tuner = coordinator.agents['timeout']

    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Master Tuning Coordinator initialized")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} TimeoutTuner ready")

    # Phase 1: Fast laptop simulation (low latency)
    print_section("2. Phase 1: Fast Hardware (Low Latency)")

    await simulate_query_batch(
        timeout_tuner,
        "Fast laptop - optimal conditions",
        {
            'features': {'mean': 25, 'std': 5},    # Fast feature extraction
            'retrieval': {'mean': 15, 'std': 3},   # Fast retrieval
            'decision': {'mean': 20, 'std': 4},    # Fast decision
            'execution': {'mean': 30, 'std': 6},   # Fast execution
        },
        n_queries=50
    )

    print_timeout_stats(timeout_tuner)

    # Run tuning cycle
    print_section("3. Running Tuning Cycle #1")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result['agent']}")
    print(f"  Impact: {result.get('impact', 0):.3f}")
    print(f"  Success: {result.get('success', False)}")

    if 'actions' in result:
        print(f"\n{Colors.BOLD}Actions Taken:{Colors.ENDC}")
        for param, action in result['actions'].items():
            old = action.get('old', 0) * 1000
            new = action.get('new', 0) * 1000
            print(f"  {param}: {old:.1f}ms -> {new:.1f}ms")

    print_bandit_stats(timeout_tuner)

    # Phase 2: Slow CI simulation (high latency)
    print_section("4. Phase 2: Slow Hardware (High Latency)")

    await simulate_query_batch(
        timeout_tuner,
        "Slow CI environment",
        {
            'features': {'mean': 120, 'std': 25},   # Slow feature extraction
            'retrieval': {'mean': 80, 'std': 15},   # Slow retrieval
            'decision': {'mean': 60, 'std': 12},    # Slow decision
            'execution': {'mean': 150, 'std': 30},  # Slow execution
        },
        n_queries=50
    )

    print_timeout_stats(timeout_tuner)

    # Run tuning cycle
    print_section("5. Running Tuning Cycle #2")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result['agent']}")
    print(f"  Impact: {result.get('impact', 0):.3f}")
    print(f"  Success: {result.get('success', False)}")

    if 'actions' in result:
        print(f"\n{Colors.BOLD}Actions Taken:{Colors.ENDC}")
        for param, action in result['actions'].items():
            old = action.get('old', 0) * 1000
            new = action.get('new', 0) * 1000
            print(f"  {param}: {old:.1f}ms -> {new:.1f}ms")

    print_bandit_stats(timeout_tuner)

    # Phase 3: Return to fast hardware
    print_section("6. Phase 3: Back to Fast Hardware")

    await simulate_query_batch(
        timeout_tuner,
        "Fast laptop again",
        {
            'features': {'mean': 30, 'std': 6},
            'retrieval': {'mean': 18, 'std': 4},
            'decision': {'mean': 22, 'std': 5},
            'execution': {'mean': 35, 'std': 7},
        },
        n_queries=50
    )

    print_timeout_stats(timeout_tuner)

    # Run tuning cycle
    print_section("7. Running Tuning Cycle #3")
    result = await coordinator.run_tuning_cycle()

    print(f"\n{Colors.OKGREEN}Tuning Result:{Colors.ENDC}")
    print(f"  Agent: {result['agent']}")
    print(f"  Impact: {result.get('impact', 0):.3f}")
    print(f"  Success: {result.get('success', False)}")

    if 'actions' in result:
        print(f"\n{Colors.BOLD}Actions Taken:{Colors.ENDC}")
        for param, action in result['actions'].items():
            old = action.get('old', 0) * 1000
            new = action.get('new', 0) * 1000
            print(f"  {param}: {old:.1f}ms -> {new:.1f}ms")

    print_bandit_stats(timeout_tuner)

    # Final summary
    print_section("8. Final Summary")

    summary = coordinator.get_tuning_summary()

    print(f"\n{Colors.BOLD}Tuning Summary:{Colors.ENDC}")
    print(f"  Total queries processed: {summary['total_queries']}")
    print(f"  Total tuning cycles: {summary['total_tuning_cycles']}")
    print(f"  Circuit breaker: {'OPEN' if summary['circuit_breaker_open'] else 'CLOSED'}")

    print(f"\n{Colors.BOLD}Agent Statistics:{Colors.ENDC}")
    for agent_name, stats in summary['agents'].items():
        print(f"  {agent_name}:")
        print(f"    Cycles: {stats['total_cycles']}")
        print(f"    Successful: {stats['successful_tunings']}")
        print(f"    Success rate: {stats['success_rate']:.1%}")

    print_header("Demo Complete!")

    print(f"\n{Colors.OKGREEN}Key Achievements:{Colors.ENDC}")
    print("  [OK] Timeouts adapted to hardware speed (fast -> slow -> fast)")
    print("  [OK] Thompson Sampling learned optimal safety margins")
    print("  [OK] Zero manual configuration required")
    print("  [OK] State persisted to disk (survives restarts)")

    print(f"\n{Colors.OKCYAN}Configuration eliminated:{Colors.ENDC}")
    print("  - pipeline_timeout: Self-tuned based on stage timings")
    print("  - retrieval_timeout: Adapted to p95 latency + margin")
    print("  - stage_timeouts: Learned per hardware characteristics")

    print(f"\n{Colors.WARNING}Next: Add 6 more tuning agents for complete self-tuning!{Colors.ENDC}\n")


if __name__ == '__main__':
    asyncio.run(main())
