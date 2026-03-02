#!/usr/bin/env python3
"""
Verification: All 5 Common Extensions

Tests that all documented extensions work correctly.
"""

import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
from datetime import time
from hololoom.context import (
    LearningTracker,
    StrategyUpdater,
    RoutingEvent
)


def print_header(text: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


# ============================================================================
# Extension 1: Query Type Detection
# ============================================================================

def classify_query_type(query: str) -> str:
    """Classify query into type"""
    query_lower = query.lower()

    if any(word in query_lower for word in ['select', 'get', 'show', 'list']):
        return 'read'
    elif any(word in query_lower for word in ['insert', 'create', 'add']):
        return 'write'
    elif any(word in query_lower for word in ['update', 'modify', 'change']):
        return 'update'
    elif any(word in query_lower for word in ['delete', 'remove']):
        return 'delete'
    else:
        return 'unknown'


def test_query_type_detection():
    """Test 1: Query Type Detection"""
    print_header("Test 1: Query Type Detection")

    queries = [
        "SELECT * FROM policies",
        "INSERT INTO policies VALUES (1, 'test')",
        "UPDATE policies SET name='new'",
        "DELETE FROM policies WHERE id=1",
        "What is the policy?"
    ]

    results = {}
    for query in queries:
        qtype = classify_query_type(query)
        results[qtype] = results.get(qtype, 0) + 1
        print(f"  '{query[:40]}...' -> {qtype}")

    print()
    print("Query type distribution:")
    for qtype, count in results.items():
        print(f"  {qtype}: {count}")

    if all(qtype in results for qtype in ['read', 'write', 'update', 'delete', 'unknown']):
        print("\n[PASS] Query type detection working")
        return True
    else:
        print("\n[FAIL] Missing query types")
        return False


# ============================================================================
# Extension 2: Cost Tracking
# ============================================================================

def test_cost_tracking():
    """Test 2: Cost Tracking"""
    print_header("Test 2: Cost Tracking")

    # Cost per backend ($ per query)
    cost_per_backend = {
        "sql": 0.001,      # $0.001 per query
        "neo4j": 0.005,    # $0.005 per query
        "qdrant": 0.002    # $0.002 per query
    }

    # Simulate routing events
    routing_history = [
        {"backend": "sql", "count": 100},
        {"backend": "neo4j", "count": 50},
        {"backend": "qdrant", "count": 75}
    ]

    # Calculate costs
    total_cost = 0.0
    cost_breakdown = {}

    for event in routing_history:
        backend = event["backend"]
        count = event["count"]
        cost = cost_per_backend[backend] * count
        cost_breakdown[backend] = cost
        total_cost += cost

    print("Cost breakdown:")
    for backend, cost in cost_breakdown.items():
        print(f"  {backend}: ${cost:.3f}")

    print(f"\nTotal cost: ${total_cost:.3f}")

    if total_cost > 0:
        print("\n[PASS] Cost tracking working")
        return True
    else:
        print("\n[FAIL] Cost calculation incorrect")
        return False


# ============================================================================
# Extension 3: A/B Testing
# ============================================================================

class ABTestingRouter:
    """Simple A/B testing router"""

    def __init__(self, split_ratio=0.5):
        self.split_ratio = split_ratio
        self.group_a_count = 0
        self.group_b_count = 0

    def route(self, query_id: int) -> str:
        """Route to A or B based on split"""
        import random
        random.seed(query_id)  # Deterministic for testing

        if random.random() < self.split_ratio:
            self.group_a_count += 1
            return "A"
        else:
            self.group_b_count += 1
            return "B"


def test_ab_testing():
    """Test 3: A/B Testing"""
    print_header("Test 3: A/B Testing")

    router = ABTestingRouter(split_ratio=0.5)

    # Route 100 queries
    for i in range(100):
        group = router.route(i)

    print(f"Group A: {router.group_a_count} queries")
    print(f"Group B: {router.group_b_count} queries")

    split_ratio = router.group_a_count / (router.group_a_count + router.group_b_count)
    print(f"Actual split: {split_ratio:.2f} (target: 0.50)")

    if 0.4 <= split_ratio <= 0.6:
        print("\n[PASS] A/B testing working (split within 40-60%)")
        return True
    else:
        print("\n[FAIL] Split ratio outside acceptable range")
        return False


# ============================================================================
# Extension 4: Time-based Rules
# ============================================================================

class TimeAwareStrategyUpdater(StrategyUpdater):
    """Strategy updater with time-of-day rules"""

    def __init__(self, query_router=None, **kwargs):
        # For testing, router can be None
        if query_router is not None:
            super().__init__(query_router, **kwargs)
        else:
            self.router = None

        # Define time-of-day rules
        self.time_rules = {
            "peak_hours": (time(9, 0), time(17, 0)),    # 9am-5pm
            "off_peak": (time(17, 0), time(9, 0)),       # 5pm-9am
        }

    def is_peak_hours(self, current_time=None) -> bool:
        """Check if current time is peak hours"""
        if current_time is None:
            from datetime import datetime
            current_time = datetime.now().time()

        start, end = self.time_rules["peak_hours"]

        if start < end:
            return start <= current_time <= end
        else:  # Wraps midnight
            return current_time >= start or current_time <= end


def test_time_based_rules():
    """Test 4: Time-based Rules"""
    print_header("Test 4: Time-based Rules")

    updater = TimeAwareStrategyUpdater()

    # Test various times
    test_times = [
        (time(8, 0), False, "8:00 AM - Off-peak"),
        (time(10, 0), True, "10:00 AM - Peak"),
        (time(12, 0), True, "12:00 PM - Peak"),
        (time(18, 0), False, "6:00 PM - Off-peak"),
        (time(23, 0), False, "11:00 PM - Off-peak"),
    ]

    results = []
    for test_time, expected, label in test_times:
        actual = updater.is_peak_hours(test_time)
        status = "[OK]" if actual == expected else "[X]"
        results.append(actual == expected)
        print(f"  {status} {label}: {actual} (expected {expected})")

    if all(results):
        print("\n[PASS] Time-based rules working")
        return True
    else:
        print("\n[FAIL] Some time checks failed")
        return False


# ============================================================================
# Extension 5: External Alerts
# ============================================================================

class AlertingLearningTracker(LearningTracker):
    """Learning tracker with alert callbacks"""

    def __init__(self, reflection_buffer=None):
        super().__init__(reflection_buffer)
        self.alerts = []

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        # Call parent
        await super().record_routing(
            session_id, query, backend,
            predicted_confidence, actual_confidence,
            latency_ms, cache_hit, fallback_used
        )

        # Check for alert conditions
        if latency_ms > 100.0:
            self.alerts.append({
                "type": "high_latency",
                "backend": backend,
                "latency_ms": latency_ms,
                "message": f"High latency detected: {latency_ms:.1f}ms on {backend}"
            })

        if actual_confidence < 0.60:
            self.alerts.append({
                "type": "low_confidence",
                "backend": backend,
                "confidence": actual_confidence,
                "message": f"Low confidence detected: {actual_confidence:.2f} on {backend}"
            })


async def test_external_alerts():
    """Test 5: External Alerts"""
    print_header("Test 5: External Alerts")

    tracker = AlertingLearningTracker()

    # Record events that should trigger alerts
    await tracker.record_routing(
        session_id="s1",
        query="Slow query",
        backend="sql",
        predicted_confidence=0.85,
        actual_confidence=0.90,
        latency_ms=150.0  # High latency!
    )

    await tracker.record_routing(
        session_id="s2",
        query="Low confidence query",
        backend="neo4j",
        predicted_confidence=0.80,
        actual_confidence=0.55,  # Low confidence!
        latency_ms=10.0
    )

    await tracker.record_routing(
        session_id="s3",
        query="Normal query",
        backend="sql",
        predicted_confidence=0.85,
        actual_confidence=0.90,
        latency_ms=15.0  # Normal
    )

    # Check alerts
    print(f"Total alerts: {len(tracker.alerts)}")
    for alert in tracker.alerts:
        print(f"  [{alert['type']}] {alert['message']}")

    # Should have 2 alerts (high latency + low confidence)
    if len(tracker.alerts) == 2:
        print("\n[PASS] External alerts working")
        return True
    else:
        print(f"\n[FAIL] Expected 2 alerts, got {len(tracker.alerts)}")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all extension verification tests"""
    print("=" * 80)
    print("Extension Verification Suite")
    print("=" * 80)
    print()
    print("Testing 5 common extensions...")
    print()

    # Run tests
    tests = [
        ("Query Type Detection", test_query_type_detection),
        ("Cost Tracking", test_cost_tracking),
        ("A/B Testing", test_ab_testing),
        ("Time-based Rules", test_time_based_rules),
        ("External Alerts", test_external_alerts),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print()
    print(f"Tests Passed: {passed}/{total}")
    print()

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print()

    if passed == total:
        print("[SUCCESS] All extensions verified!")
    else:
        print(f"[FAILED] {total - passed} extension(s) need attention")

    print()


if __name__ == "__main__":
    asyncio.run(main())
