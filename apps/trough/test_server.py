#!/usr/bin/env python3
"""
Quick test script for Squad server
"""

import requests
import json
from time import time

SERVER_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_query(text: str, mode: str = "direct"):
    """Test query endpoint"""
    print(f"Testing /query with: '{text}'")
    print(f"Mode: {mode}")

    start = time()
    response = requests.post(
        f"{SERVER_URL}/query",
        json={"text": text, "mode": mode}
    )
    duration = (time() - start) * 1000

    print(f"Status: {response.status_code}")
    print(f"Duration: {duration:.0f}ms")

    result = response.json()
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Response: {result['response']}")
    print(f"Steps: {len(result['steps_taken'])}")
    print()

    return result

def test_stats():
    """Test stats endpoint"""
    print("Testing /stats endpoint...")
    response = requests.get(f"{SERVER_URL}/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Squad Server Test Suite")
    print("=" * 60)
    print()

    try:
        # Test 1: Health
        test_health()

        # Test 2: Simple query
        test_query("What is Python?")

        # Test 3: Complex query
        test_query("Explain Thompson Sampling vs Epsilon-Greedy")

        # Test 4: Stats
        test_stats()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server.")
        print("Make sure the server is running:")
        print("  PYTHONPATH=.. python server_simple.py")
    except Exception as e:
        print(f"ERROR: {e}")
