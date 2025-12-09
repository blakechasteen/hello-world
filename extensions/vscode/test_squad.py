#!/usr/bin/env python3
"""
Squad Server Test Script
Automated testing for Squad API endpoints
"""

import sys
import time
import requests
import json
from typing import Dict, Any, List
from datetime import datetime


class SquadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def test_health(self) -> bool:
        """Test /health endpoint"""
        self.log("Testing /health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log("✅ Health check passed", "SUCCESS")
                    self.results.append({
                        "test": "health_check",
                        "status": "pass",
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    })
                    return True
                else:
                    self.log(f"❌ Health check failed: {data}", "ERROR")
                    self.results.append({"test": "health_check", "status": "fail", "error": str(data)})
                    return False
            else:
                self.log(f"❌ Health check failed with status {response.status_code}", "ERROR")
                self.results.append({"test": "health_check", "status": "fail", "error": f"HTTP {response.status_code}"})
                return False

        except requests.exceptions.ConnectionError:
            self.log("❌ Cannot connect to server", "ERROR")
            self.results.append({"test": "health_check", "status": "fail", "error": "Connection refused"})
            return False
        except Exception as e:
            self.log(f"❌ Health check error: {e}", "ERROR")
            self.results.append({"test": "health_check", "status": "fail", "error": str(e)})
            return False

    def test_query_direct(self) -> bool:
        """Test /query endpoint with DIRECT mode"""
        self.log("Testing /query endpoint (DIRECT mode)...")
        try:
            payload = {
                "text": "What is Thompson Sampling?",
                "mode": "direct",
                "max_steps": 3
            }

            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                required_fields = ["response", "confidence", "reasoning_mode", "steps_taken", "total_duration_ms"]
                missing_fields = [f for f in required_fields if f not in data]

                if missing_fields:
                    self.log(f"❌ Missing fields: {missing_fields}", "ERROR")
                    self.results.append({"test": "query_direct", "status": "fail", "error": f"Missing fields: {missing_fields}"})
                    return False

                self.log(f"✅ Query DIRECT passed (confidence: {data['confidence']:.2f}, duration: {data['total_duration_ms']:.0f}ms)", "SUCCESS")
                self.results.append({
                    "test": "query_direct",
                    "status": "pass",
                    "confidence": data['confidence'],
                    "duration_ms": data['total_duration_ms'],
                    "steps": len(data['steps_taken'])
                })
                return True
            else:
                self.log(f"❌ Query failed with status {response.status_code}", "ERROR")
                self.results.append({"test": "query_direct", "status": "fail", "error": f"HTTP {response.status_code}"})
                return False

        except Exception as e:
            self.log(f"❌ Query error: {e}", "ERROR")
            self.results.append({"test": "query_direct", "status": "fail", "error": str(e)})
            return False

    def test_query_verify(self) -> bool:
        """Test /query endpoint with VERIFY mode"""
        self.log("Testing /query endpoint (VERIFY mode)...")
        try:
            payload = {
                "text": "Explain how Thompson Sampling balances exploration and exploitation",
                "mode": "verify",
                "max_steps": 5
            }

            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Query VERIFY passed (confidence: {data['confidence']:.2f}, duration: {data['total_duration_ms']:.0f}ms)", "SUCCESS")
                self.results.append({
                    "test": "query_verify",
                    "status": "pass",
                    "confidence": data['confidence'],
                    "duration_ms": data['total_duration_ms'],
                    "steps": len(data['steps_taken'])
                })
                return True
            else:
                self.log(f"❌ Query VERIFY failed with status {response.status_code}", "ERROR")
                self.results.append({"test": "query_verify", "status": "fail", "error": f"HTTP {response.status_code}"})
                return False

        except Exception as e:
            self.log(f"❌ Query VERIFY error: {e}", "ERROR")
            self.results.append({"test": "query_verify", "status": "fail", "error": str(e)})
            return False

    def test_chat(self) -> bool:
        """Test /chat endpoint"""
        self.log("Testing /chat endpoint...")
        try:
            payload = {
                "message": "Hello, Squad!",
            }

            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    self.log(f"✅ Chat passed (confidence: {data.get('confidence', 0):.2f})", "SUCCESS")
                    self.results.append({
                        "test": "chat",
                        "status": "pass",
                        "confidence": data.get('confidence', 0)
                    })
                    return True
                else:
                    self.log("❌ Chat response missing 'response' field", "ERROR")
                    self.results.append({"test": "chat", "status": "fail", "error": "Missing response field"})
                    return False
            else:
                self.log(f"❌ Chat failed with status {response.status_code}", "ERROR")
                self.results.append({"test": "chat", "status": "fail", "error": f"HTTP {response.status_code}"})
                return False

        except Exception as e:
            self.log(f"❌ Chat error: {e}", "ERROR")
            self.results.append({"test": "chat", "status": "fail", "error": str(e)})
            return False

    def test_stats(self) -> bool:
        """Test /stats endpoint"""
        self.log("Testing /stats endpoint...")
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=5)

            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Stats passed: {data}", "SUCCESS")
                self.results.append({
                    "test": "stats",
                    "status": "pass",
                    "data": data
                })
                return True
            else:
                self.log(f"❌ Stats failed with status {response.status_code}", "ERROR")
                self.results.append({"test": "stats", "status": "fail", "error": f"HTTP {response.status_code}"})
                return False

        except Exception as e:
            self.log(f"❌ Stats error: {e}", "ERROR")
            self.results.append({"test": "stats", "status": "fail", "error": str(e)})
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and print summary"""
        self.log("=" * 60)
        self.log("Squad Server Test Suite")
        self.log("=" * 60)

        # Test 1: Health check
        health_ok = self.test_health()
        if not health_ok:
            self.log("⚠️  Server not healthy - skipping remaining tests", "WARNING")
            self.print_summary()
            return False

        time.sleep(1)

        # Test 2: Query DIRECT mode
        self.test_query_direct()
        time.sleep(1)

        # Test 3: Query VERIFY mode
        self.test_query_verify()
        time.sleep(1)

        # Test 4: Chat endpoint
        self.test_chat()
        time.sleep(1)

        # Test 5: Stats endpoint
        self.test_stats()

        # Print summary
        self.print_summary()

        # Return overall pass/fail
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        return passed == total

    def print_summary(self):
        """Print test summary"""
        self.log("=" * 60)
        self.log("Test Summary")
        self.log("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = total - passed

        self.log(f"Total:  {total} tests")
        self.log(f"Passed: {passed} tests", "SUCCESS" if passed == total else "INFO")
        self.log(f"Failed: {failed} tests", "ERROR" if failed > 0 else "INFO")

        if failed == 0:
            self.log("🎉 All tests passed!", "SUCCESS")
        else:
            self.log("❌ Some tests failed", "ERROR")
            for result in self.results:
                if result["status"] == "fail":
                    self.log(f"  - {result['test']}: {result.get('error', 'Unknown error')}", "ERROR")

        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"Results saved to test_results.json")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Squad Server Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    args = parser.parse_args()

    tester = SquadTester(base_url=args.url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
