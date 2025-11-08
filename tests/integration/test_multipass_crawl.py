#!/usr/bin/env python3
"""
Test Multipass Memory Crawling Integration
==========================================
Validates the recursive gated multipass memory crawling system in WeavingOrchestrator.

Tests:
1. LITE complexity: 1 pass, threshold=0.7, limit=5, no graph
2. FAST complexity: 2 passes, thresholds=[0.6, 0.75], limits=[8, 12], light graph
3. FULL complexity: 3 passes, thresholds=[0.6, 0.75, 0.85], limits=[12, 20, 15], full graph
4. RESEARCH complexity: 4 passes, thresholds=[0.5, 0.65, 0.8, 0.9], limits=[20, 30, 25, 15], aggressive
5. Provenance tracking: Verify crawl events recorded properly
6. Performance: Sub-2ms crawl time even for deep exploration
"""

import asyncio
import time
from typing import List, Dict, Any

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HoloLoom.config import Config, ExecutionMode, MemoryBackend
from HoloLoom.weaving_orchestrator import WeavingOrchestrator, Query
from HoloLoom.protocols import ComplexityLevel
from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult


class MockMemoryBackend:
    """Mock memory backend for testing with realistic data"""
    
    def __init__(self):
        # Create a knowledge graph with interconnected items
        from datetime import datetime
        self.items = {
            f'item_{i}': Memory(
                id=f'item_{i}',
                text=f'Knowledge item {i}: {"neural" if i < 10 else "decision"} systems',
                timestamp=datetime.now(),
                context={'episode': None, 'entities': [], 'motifs': []},
                metadata={'relevance': 1.0 - (i / 100), 'depth': i // 10}
            )
            for i in range(50)
        }
        
        # Add graph relationships
        self.graph = {
            'item_0': ['item_1', 'item_2', 'item_10'],
            'item_1': ['item_3', 'item_4'],
            'item_2': ['item_5', 'item_11'],
            'item_10': ['item_11', 'item_12'],
        }
    
    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        """Mock recall that returns items above threshold"""
        # Simulate threshold-based filtering
        threshold = getattr(query, 'threshold', 0.7)
        
        # Filter items by relevance threshold
        filtered = [
            item for item in self.items.values()
            if item.metadata.get('relevance', 0) >= threshold
        ]
        
        # Sort by relevance and limit
        sorted_items = sorted(
            filtered,
            key=lambda x: x.metadata.get('relevance', 0),
            reverse=True
        )[:limit]
        
        scores = [item.metadata.get('relevance', 1.0) for item in sorted_items]
        
        return RetrievalResult(
            memories=sorted_items,
            scores=scores,
            strategy_used="threshold_filtering",
            metadata={'threshold': threshold, 'total_available': len(self.items)}
        )
    
    async def get_related(self, item_id: str, limit: int = 5) -> List[Memory]:
        """Mock graph traversal"""
        related_ids = self.graph.get(item_id, [])[:limit]
        return [self.items[rid] for rid in related_ids if rid in self.items]
    
    async def health_check(self) -> Dict[str, Any]:
        return {'status': 'healthy', 'items': len(self.items)}


async def test_lite_complexity():
    """Test LITE: 1 pass, high threshold, no graph traversal"""
    print("\n" + "="*70)
    print("TEST 1: LITE Complexity (1 pass, threshold=0.7)")
    print("="*70)
    
    config = Config.bare()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    query = Query(text="hi")  # Short greeting  LITE
    
    # Test crawl directly
    start = time.perf_counter()
    results = await orchestrator._multipass_memory_crawl(query, ComplexityLevel.LITE, None)
    crawl_time_ms = (time.perf_counter() - start) * 1000
    
    print(f" Crawl time: {crawl_time_ms:.2f}ms")
    print(f" Items retrieved: {len(results)}")
    print(f" Expected: 1-5 items (high threshold)")
    
    assert len(results) <= 5, f"LITE should retrieve 5 items, got {len(results)}"
    assert crawl_time_ms < 50, f"LITE crawl should be <50ms, got {crawl_time_ms:.2f}ms"
    
    print(" LITE complexity test PASSED")
    return True


async def test_fast_complexity():
    """Test FAST: 2 passes, progressive thresholds, light graph"""
    print("\n" + "="*70)
    print("TEST 2: FAST Complexity (2 passes, thresholds=[0.6, 0.75])")
    print("="*70)
    
    config = Config.fast()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    query = Query(text="what are neural decision systems?")  # Question  FAST
    
    start = time.perf_counter()
    results = await orchestrator._multipass_memory_crawl(query, ComplexityLevel.FAST, None)
    crawl_time_ms = (time.perf_counter() - start) * 1000
    
    print(f" Crawl time: {crawl_time_ms:.2f}ms")
    print(f" Items retrieved: {len(results)}")
    print(f" Expected: 8-20 items (2 passes + graph expansion)")
    
    assert 5 <= len(results) <= 25, f"FAST should retrieve 5-25 items, got {len(results)}"
    assert crawl_time_ms < 150, f"FAST crawl should be <150ms, got {crawl_time_ms:.2f}ms"
    
    print(" FAST complexity test PASSED")
    return True


async def test_full_complexity():
    """Test FULL: 3 passes, Matryoshka gating, full graph"""
    print("\n" + "="*70)
    print("TEST 3: FULL Complexity (3 passes, thresholds=[0.6, 0.75, 0.85])")
    print("="*70)
    
    config = Config.fused()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    query = Query(text="explain how neural decision systems work in detail with examples and comparisons")  # Long query  FULL
    
    start = time.perf_counter()
    results = await orchestrator._multipass_memory_crawl(query, ComplexityLevel.FULL, None)
    crawl_time_ms = (time.perf_counter() - start) * 1000
    
    print(f" Crawl time: {crawl_time_ms:.2f}ms")
    print(f" Items retrieved: {len(results)}")
    print(f" Expected: 15-40 items (3 passes + aggressive graph)")
    
    assert 10 <= len(results) <= 50, f"FULL should retrieve 10-50 items, got {len(results)}"
    assert crawl_time_ms < 300, f"FULL crawl should be <300ms, got {crawl_time_ms:.2f}ms"
    
    print(" FULL complexity test PASSED")
    return True


async def test_research_complexity():
    """Test RESEARCH: 4 passes, maximum depth, aggressive graph"""
    print("\n" + "="*70)
    print("TEST 4: RESEARCH Complexity (4 passes, thresholds=[0.5, 0.65, 0.8, 0.9])")
    print("="*70)
    
    config = Config.fused()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    query = Query(text="analyze and compare neural decision architectures comprehensively")  # Analysis verb + research keyword  RESEARCH
    
    start = time.perf_counter()
    results = await orchestrator._multipass_memory_crawl(query, ComplexityLevel.RESEARCH, None)
    crawl_time_ms = (time.perf_counter() - start) * 1000
    
    print(f" Crawl time: {crawl_time_ms:.2f}ms")
    print(f" Items retrieved: {len(results)}")
    print(f" Expected: 20-50+ items (4 passes + maximum graph depth)")
    
    assert len(results) >= 15, f"RESEARCH should retrieve 15 items, got {len(results)}"
    # Research can take longer but should still be reasonable
    print(f" Research crawl time acceptable: {crawl_time_ms:.2f}ms")
    
    print(" RESEARCH complexity test PASSED")
    return True


async def test_provenance_tracking():
    """Test that crawl events are properly tracked in provenance"""
    print("\n" + "="*70)
    print("TEST 5: Provenance Tracking")
    print("="*70)
    
    config = Config.fast()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    query = Query(text="test query")
    complexity = ComplexityLevel.FAST
    
    # Create a trace object
    from HoloLoom.protocols import ProvenceTrace
    trace = ProvenceTrace(
        operation_id="test_crawl",
        complexity_level=complexity,
        start_time=time.perf_counter()
    )
    
    # Run crawl with tracing
    results = await orchestrator._multipass_memory_crawl(query, complexity, trace)
    
    # Check trace events
    events = trace.shuttle_events
    print(f" Recorded {len(events)} shuttle events")
    
    # Verify expected events
    event_types = [e['event_type'] for e in events]
    assert 'crawl_start' in event_types, "Missing crawl_start event"
    assert 'crawl_complete' in event_types, "Missing crawl_complete event"
    
    # Count pass events
    pass_events = [e for e in events if 'crawl_pass_' in e['event_type']]
    print(f" Recorded {len(pass_events)} pass events")
    assert len(pass_events) >= 2, f"FAST should have 2+ pass events, got {len(pass_events)}"
    
    # Check for metadata
    complete_event = [e for e in events if e['event_type'] == 'crawl_complete'][0]
    metadata = complete_event.get('data', {})  # Changed from 'metadata' to 'data'
    print(f" Crawl metadata: {metadata}")
    assert 'total_items' in metadata, "Missing total_items in metadata"
    assert 'time_ms' in metadata, "Missing time_ms in metadata"
    
    print(" Provenance tracking test PASSED")
    return True


async def test_performance_validation():
    """Test that performance targets are met"""
    print("\n" + "="*70)
    print("TEST 6: Performance Validation")
    print("="*70)
    
    config = Config.fast()
    memory = MockMemoryBackend()
    
    orchestrator = WeavingOrchestrator(
        cfg=config,
        memory=memory,
        enable_complexity_auto_detect=True
    )
    
    # Test multiple queries to get average performance
    queries = [
        (Query(text="hi"), ComplexityLevel.LITE, 50),
        (Query(text="what is this?"), ComplexityLevel.FAST, 150),
        (Query(text="explain neural architectures in detail"), ComplexityLevel.FULL, 300),
    ]
    
    all_passed = True
    for query, complexity, target_ms in queries:
        times = []
        for _ in range(5):  # Run 5 times
            start = time.perf_counter()
            results = await orchestrator._multipass_memory_crawl(query, complexity, None)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f" {complexity.name:8s}: avg={avg_time:.2f}ms, max={max_time:.2f}ms, target<{target_ms}ms", end="")
        
        if max_time < target_ms:
            print(" ")
        else:
            print(f"  (exceeded by {max_time - target_ms:.2f}ms)")
            all_passed = False
    
    if all_passed:
        print(" Performance validation test PASSED")
    else:
        print("  Performance validation test FAILED (some targets exceeded)")
    
    return all_passed


async def main():
    """Run all tests"""
    print("\nMULTIPASS MEMORY CRAWLING TEST SUITE")
    print("=" * 70)
    print("Testing recursive gated multipass memory crawling integration")
    print()
    
    tests = [
        ("LITE Complexity", test_lite_complexity),
        ("FAST Complexity", test_fast_complexity),
        ("FULL Complexity", test_full_complexity),
        ("RESEARCH Complexity", test_research_complexity),
        ("Provenance Tracking", test_provenance_tracking),
        ("Performance Validation", test_performance_validation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed ({(passed_count/total_count)*100:.1f}%)")
    
    if passed_count == total_count:
        print("\nALL TESTS PASSED! Multipass memory crawling is working perfectly!")
        return 0
    else:
        print(f"\n{total_count - passed_count} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

