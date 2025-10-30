"""
Tests for Learned Routing System

Validates Thompson Sampling, metrics collection, A/B testing, and integration.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.routing.learned import ThompsonBandit, LearnedRouter
from HoloLoom.routing.metrics import RoutingMetrics, MetricsCollector
from HoloLoom.routing.ab_test import ABTestRouter, StrategyVariant
from HoloLoom.routing.integration import RoutingOrchestrator, classify_query_type, rule_based_routing


def test_thompson_bandit():
    """Test Thompson Sampling bandit."""
    print("\n=== Test 1: Thompson Sampling Bandit ===")
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    bandit = ThompsonBandit(backends=backends)
    
    # Initial state
    print("Initial stats:")
    stats = bandit.get_stats()
    for backend, backend_stats in stats.items():
        print(f"  {backend}: {backend_stats['expected_success_rate']:.3f} "
              f"(alpha={backend_stats['alpha']:.1f}, beta={backend_stats['beta']:.1f})")
    
    # Simulate outcomes - HYBRID performs best
    outcomes = [
        ('HYBRID', True),
        ('HYBRID', True),
        ('HYBRID', True),
        ('NEO4J_QDRANT', True),
        ('NEO4J_QDRANT', False),
        ('NETWORKX', False),
        ('NETWORKX', False),
    ]
    
    for backend, success in outcomes:
        bandit.update(backend, success)
    
    # After learning
    print("\nAfter 7 observations:")
    stats = bandit.get_stats()
    for backend, backend_stats in stats.items():
        print(f"  {backend}: {backend_stats['expected_success_rate']:.3f} "
              f"(observations={backend_stats['total_observations']})")
    
    # Test selection - should heavily favor HYBRID
    selections = {}
    for _ in range(100):
        backend = bandit.select()
        selections[backend] = selections.get(backend, 0) + 1
    
    print("\nSelection distribution (100 samples):")
    for backend, count in sorted(selections.items(), key=lambda x: -x[1]):
        print(f"  {backend}: {count}%")
    
    # Verify HYBRID is selected most often
    assert selections['HYBRID'] > 50, "HYBRID should be selected most often"
    
    print("PASSED")


def test_learned_router():
    """Test learned router with per-query-type bandits."""
    print("\n=== Test 2: Learned Router ===")
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    query_types = ['factual', 'analytical', 'creative', 'conversational']
    
    # Create temporary storage
    storage_path = Path(__file__).parent / 'test_bandit_params.json'
    
    router = LearnedRouter(
        backends=backends,
        query_types=query_types,
        storage_path=storage_path
    )
    
    # Simulate learning:
    # - Factual queries: NETWORKX works best (fast lookups)
    # - Analytical queries: HYBRID works best (deep analysis)
    # - Creative queries: NEO4J_QDRANT works best (graph exploration)
    
    learning_data = [
        ('factual', 'NETWORKX', True),
        ('factual', 'NETWORKX', True),
        ('factual', 'HYBRID', False),
        ('analytical', 'HYBRID', True),
        ('analytical', 'HYBRID', True),
        ('analytical', 'NETWORKX', False),
        ('creative', 'NEO4J_QDRANT', True),
        ('creative', 'NEO4J_QDRANT', True),
        ('creative', 'NETWORKX', False),
    ]
    
    for query_type, backend, success in learning_data:
        router.update(query_type, backend, success)
    
    # Test routing
    print("\nLearned preferences:")
    stats = router.get_stats()
    
    for query_type in ['factual', 'analytical', 'creative']:
        print(f"\n{query_type.capitalize()}:")
        backend_stats = stats[query_type]
        
        # Sort by expected success rate
        sorted_backends = sorted(
            backend_stats.items(),
            key=lambda x: x[1]['expected_success_rate'],
            reverse=True
        )
        
        for backend, backend_data in sorted_backends:
            if backend_data['total_observations'] > 0:
                print(f"  {backend}: {backend_data['expected_success_rate']:.3f} "
                      f"(observations={backend_data['total_observations']})")
    
    # Cleanup
    if storage_path.exists():
        storage_path.unlink()
    
    print("\nPASSED")


def test_metrics_collection():
    """Test metrics collection."""
    print("\n=== Test 3: Metrics Collection ===")
    
    # Create temporary storage
    storage_path = Path(__file__).parent / 'test_metrics.jsonl'
    
    collector = MetricsCollector(storage_path=storage_path)
    collector.clear()  # Start fresh
    
    # Record some metrics
    test_metrics = [
        RoutingMetrics(
            query="What is Python?",
            query_type="factual",
            complexity="LITE",
            backend_selected="NETWORKX",
            latency_ms=50.0,
            relevance_score=0.9,
            confidence=0.85,
            success=True,
            memory_size=10
        ),
        RoutingMetrics(
            query="Analyze this code",
            query_type="analytical",
            complexity="FULL",
            backend_selected="HYBRID",
            latency_ms=150.0,
            relevance_score=0.95,
            confidence=0.9,
            success=True,
            memory_size=50
        ),
        RoutingMetrics(
            query="Failed query",
            query_type="factual",
            complexity="FAST",
            backend_selected="NETWORKX",
            latency_ms=75.0,
            relevance_score=0.3,
            confidence=0.4,
            success=False,
            memory_size=20
        )
    ]
    
    for metrics in test_metrics:
        collector.record(metrics)
    
    # Test filtering
    factual_metrics = collector.get_metrics(query_type='factual')
    print(f"\nFactual queries: {len(factual_metrics)}")
    assert len(factual_metrics) == 2
    
    # Test backend stats
    stats = collector.get_backend_stats('NETWORKX')
    print(f"\nNETWORKX stats:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Avg relevance: {stats['avg_relevance']:.2f}")
    print(f"  Success rate: {stats['success_rate']:.2f}")
    
    assert stats['count'] == 2
    assert stats['success_rate'] == 0.5  # 1 success, 1 failure
    
    # Cleanup
    collector.clear()
    
    print("\nPASSED")


def test_ab_testing():
    """Test A/B testing framework."""
    print("\n=== Test 4: A/B Testing Framework ===")
    
    # Create temporary storage
    storage_path = Path(__file__).parent / 'test_ab_results.json'
    
    # Define strategy variants
    def learned_strategy(query_type: str, complexity: str) -> str:
        """Simulated learned strategy - always picks HYBRID."""
        return 'HYBRID'
    
    variants = [
        StrategyVariant(
            name='learned',
            weight=0.5,
            strategy_fn=learned_strategy
        ),
        StrategyVariant(
            name='rule_based',
            weight=0.5,
            strategy_fn=rule_based_routing
        )
    ]
    
    router = ABTestRouter(variants=variants, storage_path=storage_path)
    
    # Simulate queries with outcomes
    # Learned strategy performs better
    for _ in range(20):
        backend, variant = router.route('analytical', 'FULL')
        
        if variant == 'learned':
            # Learned strategy has better outcomes
            router.record_outcome(
                variant_name=variant,
                latency_ms=100.0,
                relevance_score=0.9,
                success=True
            )
        else:
            # Rule-based has worse outcomes
            router.record_outcome(
                variant_name=variant,
                latency_ms=150.0,
                relevance_score=0.7,
                success=True
            )
    
    # Check results
    results = router.get_results()
    print("\nA/B Test Results:")
    
    for variant_name, metrics in results.items():
        print(f"\n{variant_name}:")
        print(f"  Queries: {metrics['total_queries']}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  Avg relevance: {metrics['avg_relevance']:.2f}")
        print(f"  Success rate: {metrics['success_rate']:.2f}")
    
    # Get winner
    winner = router.get_winner()
    print(f"\nWinner: {winner}")
    
    assert winner == 'learned', "Learned strategy should win"
    
    # Cleanup
    if storage_path.exists():
        storage_path.unlink()
    
    print("\nPASSED")


def test_query_classification():
    """Test query type classification."""
    print("\n=== Test 5: Query Classification ===")
    
    test_cases = [
        ("What is Python?", "factual"),
        ("Who is the president?", "factual"),
        ("Why does this happen?", "analytical"),
        ("How do I solve this?", "analytical"),
        ("Imagine a future world", "creative"),
        ("Hello there!", "conversational"),
    ]
    
    print("\nQuery classifications:")
    for query, expected in test_cases:
        actual = classify_query_type(query)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {query[:30]:30s} -> {actual:15s} [{status}]")
        assert actual == expected
    
    print("\nPASSED")


def test_routing_orchestrator():
    """Test full routing orchestrator integration."""
    print("\n=== Test 6: Routing Orchestrator ===")
    
    # Create temporary storage directory
    storage_dir = Path(__file__).parent / 'test_routing'
    storage_dir.mkdir(exist_ok=True)
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    query_types = ['factual', 'analytical', 'creative', 'conversational']
    
    orchestrator = RoutingOrchestrator(
        backends=backends,
        query_types=query_types,
        enable_ab_test=True,
        storage_dir=storage_dir
    )
    
    # Simulate queries
    queries = [
        ("What is HoloLoom?", "LITE"),
        ("Analyze this complex pattern", "FULL"),
        ("Imagine a creative solution", "FAST"),
    ]
    
    print("\nSimulating queries:")
    for query, complexity in queries:
        # Select backend
        backend, strategy = orchestrator.select_backend(query, complexity)
        print(f"  {query[:30]:30s} -> {backend:15s} ({strategy})")
        
        # Simulate outcome
        orchestrator.record_outcome(
            query=query,
            complexity=complexity,
            backend_selected=backend,
            strategy_used=strategy,
            latency_ms=100.0,
            relevance_score=0.85,
            confidence=0.8,
            success=True,
            memory_size=25
        )
    
    # Get stats
    stats = orchestrator.get_routing_stats()
    print(f"\nTotal queries processed: {stats['total_queries']}")
    
    if 'ab_test' in stats:
        print("\nA/B Test Status:")
        for variant, metrics in stats['ab_test']['results'].items():
            print(f"  {variant}: {metrics['total_queries']} queries")
    
    # Cleanup
    import shutil
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    
    print("\nPASSED")


def run_all_tests():
    """Run all routing tests."""
    print("=" * 60)
    print("LEARNED ROUTING SYSTEM TESTS")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        test_thompson_bandit,
        test_learned_router,
        test_metrics_collection,
        test_ab_testing,
        test_query_classification,
        test_routing_orchestrator,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    elapsed = (time.time() - start_time) * 1000
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print(f"Time: {elapsed:.1f}ms")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
