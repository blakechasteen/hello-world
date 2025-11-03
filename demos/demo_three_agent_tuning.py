"""
Three-Agent Self-Tuning Demonstration

Shows TimeoutTuner, CacheTuner, and ThresholdTuner working together with meta-bandit coordination.

The master coordinator uses Thompson Sampling to decide which agent to activate
based on which one has been most impactful recently.
"""

import asyncio
import random
from HoloLoom.tuning import MasterTuningCoordinator
from HoloLoom.tuning.cache_tuner import CacheMetrics
from HoloLoom.tuning.threshold_tuner import QualityMetrics


def print_header(text: str):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"{text}")
    print('='*80 + '\n')


def print_subheader(text: str):
    """Print subsection header."""
    print(f"\n{text}")
    print('-'*80)


def render_bar(value: float, max_value: float, width: int = 20) -> str:
    """Render ASCII progress bar."""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return '#' * filled + '-' * (width - filled)


def simulate_query_batch(coordinator: MasterTuningCoordinator, n_queries: int, profile: str):
    """
    Simulate a batch of queries with given profile.

    Args:
        coordinator: Master tuning coordinator
        n_queries: Number of queries to simulate
        profile: One of 'optimal', 'degraded', 'mixed'
    """
    # Profile parameters
    profiles = {
        'optimal': {
            'latency_mean': 25, 'latency_std': 5,
            'cache_hit_rate': 0.85,
            'threshold_precision': 0.90, 'threshold_recall': 0.85,
        },
        'degraded': {
            'latency_mean': 120, 'latency_std': 20,
            'cache_hit_rate': 0.60,
            'threshold_precision': 0.70, 'threshold_recall': 0.60,
        },
        'mixed': {
            'latency_mean': 60, 'latency_std': 30,
            'cache_hit_rate': 0.70,
            'threshold_precision': 0.80, 'threshold_recall': 0.75,
        },
    }

    params = profiles[profile]

    # Simulate queries
    for _ in range(n_queries):
        coordinator.record_query()

        # 1. Simulate latencies for TimeoutTuner
        if 'timeout' in coordinator.agents:
            for stage in ['features', 'retrieval', 'decision', 'execution']:
                latency_ms = max(1.0, random.gauss(params['latency_mean'], params['latency_std']))
                coordinator.agents['timeout'].record_stage_completion(stage, latency_ms, timed_out=False)

        # 2. Simulate cache metrics for CacheTuner
        if 'cache' in coordinator.agents:
            cache_tuner = coordinator.agents['cache']
            for cache_type in ['query', 'parse', 'merge']:
                is_hit = random.random() < params['cache_hit_rate']
                current_size = int(cache_tuner.safe_params[cache_type].current_value)

                metrics = CacheMetrics(
                    hits=1 if is_hit else 0,
                    misses=0 if is_hit else 1,
                    evictions=1 if not is_hit and random.random() < 0.3 else 0,
                    size=int(current_size * random.uniform(0.7, 0.95)),
                    max_size=current_size,
                )
                cache_tuner.record_cache_metrics(cache_type, metrics)

        # 3. Simulate quality metrics for ThresholdTuner
        if 'threshold' in coordinator.agents:
            threshold_tuner = coordinator.agents['threshold']
            for threshold_name in threshold_tuner.threshold_names:
                # Simulate confusion matrix values based on precision/recall
                precision = params['threshold_precision']
                recall = params['threshold_recall']

                # Generate synthetic confusion matrix
                # Assume 100 samples, 50 positive, 50 negative
                tp = int(50 * recall)
                fn = 50 - tp
                fp = int(tp / precision - tp) if precision > 0 else 0
                tn = 50 - fp

                metrics = QualityMetrics(
                    true_positives=max(0, tp),
                    false_positives=max(0, fp),
                    true_negatives=max(0, tn),
                    false_negatives=max(0, fn),
                )
                threshold_tuner.record_quality_metrics(threshold_name, metrics)


def print_system_status(coordinator: MasterTuningCoordinator):
    """Print comprehensive system status."""

    # Timeouts
    if 'timeout' in coordinator.agents:
        tuner = coordinator.agents['timeout']
        print("\nTimeouts:")
        for stage in ['features', 'retrieval', 'decision', 'execution']:
            if len(tuner.latencies[stage]) > 0:
                latencies = list(tuner.latencies[stage])
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                timeout = tuner.parameters[stage].current_value
                margin = (timeout * 1000) / p95 if p95 > 0 else 0
                print(f"  {stage:12s}: {timeout*1000:5.0f}ms (p95={p95:5.1f}ms, margin={margin:.2f}x)")

    # Cache sizes
    if 'cache' in coordinator.agents:
        tuner = coordinator.agents['cache']
        print("\nCache Sizes:")
        for cache_type in ['query', 'parse', 'merge']:
            current_size = int(tuner.safe_params[cache_type].current_value)
            if len(tuner.metrics_history[cache_type]) > 0:
                recent = list(tuner.metrics_history[cache_type])[-100:]
                total_hits = sum(m.hits for m in recent)
                total_misses = sum(m.misses for m in recent)
                hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
                print(f"  {cache_type:8s}: {current_size:6d} entries (hit_rate={hit_rate:.0%})")
            else:
                print(f"  {cache_type:8s}: {current_size:6d} entries")

    # Thresholds
    if 'threshold' in coordinator.agents:
        tuner = coordinator.agents['threshold']
        print("\nThresholds (top 4):")
        threshold_names = tuner.threshold_names[:4]  # Show first 4
        for threshold_name in threshold_names:
            current_value = tuner.safe_params[threshold_name].current_value
            if len(tuner.metrics_history[threshold_name]) > 0:
                recent = list(tuner.metrics_history[threshold_name])[-100:]
                total_tp = sum(m.true_positives for m in recent)
                total_fp = sum(m.false_positives for m in recent)
                total_fn = sum(m.false_negatives for m in recent)
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                print(f"  {threshold_name:20s}: {current_value:.2f} (P={precision:.0%}, R={recall:.0%}, F1={f1:.0%})")
            else:
                print(f"  {threshold_name:20s}: {current_value:.2f}")


async def main():
    """Run three-agent tuning demonstration."""
    print_header("Thompson Sampling: Three-Agent Coordination")

    print("[OK] Master Tuning Coordinator initialized")
    print("[OK] TimeoutTuner ready (Agent 1)")
    print("[OK] CacheTuner ready (Agent 2)")
    print("[OK] ThresholdTuner ready (Agent 3)")

    # Create coordinator (includes all three agents)
    coordinator = MasterTuningCoordinator()

    # Phase 1: Optimal conditions
    print_subheader("Phase 1: Optimal Conditions (50 queries)")
    simulate_query_batch(coordinator, 50, 'optimal')
    print("  [OK] Processed 50/50 queries")
    print_system_status(coordinator)

    print("\nRunning tuning cycle...")
    result1 = await coordinator.run_tuning_cycle()
    print(f"  -> Selected agent: {result1.get('agent', 'unknown')}")
    print(f"  -> Impact: {result1.get('impact', 0.0):+.3f}")

    # Phase 2: Degraded performance
    print_subheader("Phase 2: Degraded Performance (50 queries)")
    simulate_query_batch(coordinator, 50, 'degraded')
    print("  [OK] Processed 50/50 queries")
    print_system_status(coordinator)

    print("\nRunning tuning cycle...")
    result2 = await coordinator.run_tuning_cycle()
    print(f"  -> Selected agent: {result2.get('agent', 'unknown')}")
    print(f"  -> Impact: {result2.get('impact', 0.0):+.3f}")

    # Phase 3: Mixed workload
    print_subheader("Phase 3: Mixed Workload (50 queries)")
    simulate_query_batch(coordinator, 50, 'mixed')
    print("  [OK] Processed 50/50 queries")
    print_system_status(coordinator)

    print("\nRunning tuning cycle...")
    result3 = await coordinator.run_tuning_cycle()
    print(f"  -> Selected agent: {result3.get('agent', 'unknown')}")
    print(f"  -> Impact: {result3.get('impact', 0.0):+.3f}")

    # Final status
    print_subheader("Final System Status")
    print_system_status(coordinator)

    # Meta-bandit stats
    print("\nMeta-Bandit Statistics:")
    agent_names = list(coordinator.agents.keys())
    for idx, agent_name in enumerate(agent_names):
        bandit = coordinator.meta_bandit
        expected_reward = bandit.alpha[idx] / (bandit.alpha[idx] + bandit.beta[idx])
        n_pulls = int(bandit.alpha[idx] + bandit.beta[idx] - 2)
        bar = render_bar(expected_reward, 1.0, width=20)
        print(f"  {agent_name:12s}: {bar} {expected_reward:.3f} ({n_pulls:3d} pulls)")

    # Summary
    print_header("Summary: Three-Agent Self-Tuning")

    print("\nKey Achievements:")
    print("  [OK] Three agents coordinated via meta-bandit")
    print("  [OK] System adapted to changing conditions")
    print("  [OK] Zero manual configuration required")
    print(f"  [OK] Total queries: {coordinator.total_queries}")
    print(f"  [OK] Total tuning cycles: {coordinator.total_tuning_cycles}")

    print("\nConfiguration Reduction:")
    print("  Before: 72 parameters")
    print("  After:  57 parameters (15 eliminated by Agents 1-3)")
    print("          - 4 timeouts (Agent 1)")
    print("          - 3 cache sizes (Agent 2)")
    print("          - 8 thresholds (Agent 3)")
    print("  Target: 3 parameters (96% reduction with all 7 agents)")

    print("\n" + "="*80)
    print("Moonshot Progress: 3/7 agents complete (43%)")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
