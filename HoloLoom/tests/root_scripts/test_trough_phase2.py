#!/usr/bin/env python3
"""
Test Trough Phase 2 Improvements
=================================

Tests rate limiting, query validation, and stats tracking.

Usage:
    python test_trough_phase2.py
"""

import requests
import time
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test 1: Health check (should always work, no rate limiting)."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Health check failed"
    assert response.json()["status"] == "ok", "Health status not OK"
    print("✅ PASS: Health check working")


def test_query_validation():
    """Test 2: Query size validation (should reject 200KB query)."""
    print("\n" + "="*60)
    print("TEST 2: Query Size Validation")
    print("="*60)

    # Create a 200KB query (exceeds 100KB limit)
    large_query = "x" * 200_000

    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "text": large_query,
            "mode": "direct"
        }
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)[:500]}...")

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✅ PASS: Query size validation working")


def test_max_steps_validation():
    """Test 3: max_steps validation (should reject invalid values)."""
    print("\n" + "="*60)
    print("TEST 3: max_steps Validation")
    print("="*60)

    # Try max_steps = 100 (exceeds max of 20)
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "text": "test",
            "mode": "direct",
            "max_steps": 100
        }
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✅ PASS: max_steps validation working")


def test_stats_tracking():
    """Test 4: Stats tracking (should show accurate metrics)."""
    print("\n" + "="*60)
    print("TEST 4: Stats Tracking")
    print("="*60)

    # Get initial stats
    response = requests.get(f"{BASE_URL}/stats")
    initial_stats = response.json()
    print(f"Initial total queries: {initial_stats.get('total_queries', 0)}")

    # Make 5 small queries
    for i in range(5):
        requests.post(
            f"{BASE_URL}/query",
            json={
                "text": f"test query {i}",
                "mode": "direct"
            }
        )
        time.sleep(0.1)  # Small delay to avoid rate limiting

    # Get updated stats
    response = requests.get(f"{BASE_URL}/stats")
    final_stats = response.json()

    print(f"\nFinal Stats:")
    print(json.dumps(final_stats, indent=2))

    # Verify stats updated
    initial_total = initial_stats.get('total_queries', 0)
    final_total = final_stats.get('total_queries', 0)

    assert final_total >= initial_total + 5, f"Stats not tracking correctly: {initial_total} → {final_total}"
    assert 'uptime_seconds' in final_stats, "Missing uptime_seconds"
    assert 'avg_latency_ms' in final_stats, "Missing avg_latency_ms"
    assert 'success_rate' in final_stats, "Missing success_rate"

    print(f"✅ PASS: Stats tracking working ({initial_total} → {final_total} queries)")


def test_rate_limiting():
    """Test 5: Rate limiting (should block after 60 requests)."""
    print("\n" + "="*60)
    print("TEST 5: Rate Limiting")
    print("="*60)

    print("Sending 65 requests rapidly (limit is 60/min)...")

    successes = 0
    rate_limited = 0

    for i in range(65):
        response = requests.post(
            f"{BASE_URL}/query",
            json={
                "text": f"test {i}",
                "mode": "direct"
            }
        )

        if response.status_code == 200:
            successes += 1
        elif response.status_code == 429:
            rate_limited += 1
            if rate_limited == 1:
                print(f"Rate limited after {successes} requests")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Unexpected status: {response.status_code}")

    print(f"\nResults:")
    print(f"  Successful: {successes}")
    print(f"  Rate limited: {rate_limited}")

    assert rate_limited > 0, "Rate limiting not working (expected some 429s)"
    assert successes <= 60, f"Too many successes: {successes} (expected ≤60)"
    print("✅ PASS: Rate limiting working")


def test_error_response_format():
    """Test 6: Error response format (should include retry info)."""
    print("\n" + "="*60)
    print("TEST 6: Error Response Format")
    print("="*60)

    # Trigger a validation error
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "text": "",  # Empty text should fail validation
            "mode": "direct"
        }
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    assert "detail" in response.json(), "Missing error detail"
    print("✅ PASS: Error response format correct")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TROUGH PHASE 2 TEST SUITE")
    print("="*60)

    tests = [
        test_health_check,
        test_query_validation,
        test_max_steps_validation,
        test_stats_tracking,
        test_error_response_format,
        test_rate_limiting,  # Run this last (blocks further requests)
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED! Phase 2 is production ready! 🐷")
    else:
        print(f"\n❌ {failed} test(s) failed. Check server logs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server at http://localhost:8000")
        print("   Make sure the server is running:")
        print("   PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000")
