#!/usr/bin/env python3
"""
Phase 3 Automated Test Suite
=============================
Tests Orchestrator Pipeline Visualizer and Policy & Bandit Monitor dashboards.

Usage:
    python test_phase3.py

Prerequisites:
    - Server running on localhost:8000
    - Browser installed (for Selenium tests)
"""

import sys
import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Phase3Tester:
    """Automated test suite for Phase 3 dashboards."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def print_header(self, text: str):
        """Print colored header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

    def print_test(self, test_name: str, status: str, message: str = ""):
        """Print test result."""
        if status == "PASS":
            symbol = "✓"
            color = Colors.OKGREEN
            self.tests_passed += 1
        elif status == "FAIL":
            symbol = "✗"
            color = Colors.FAIL
            self.tests_failed += 1
        else:  # WARN
            symbol = "⚠"
            color = Colors.WARNING

        print(f"{color}{symbol} {test_name}{Colors.ENDC}")
        if message:
            print(f"  {message}")

        self.test_results.append({
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    def check_server_health(self) -> bool:
        """Test 1: Server health check."""
        self.print_header("Test 1: Server Health Check")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                self.print_test("Server Health", "PASS", f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.print_test("Server Health", "FAIL", f"HTTP {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            self.print_test("Server Health", "FAIL", f"Connection error: {e}")
            return False

    def test_orchestrator_endpoint(self) -> Dict:
        """Test 2: Orchestrator monitoring endpoint."""
        self.print_header("Test 2: Orchestrator Monitoring Endpoint")

        try:
            response = requests.get(f"{self.base_url}/monitor/orchestrator", timeout=5)

            if response.status_code != 200:
                self.print_test("Orchestrator Endpoint", "FAIL", f"HTTP {response.status_code}")
                return {}

            data = response.json()

            # Check required fields
            required_fields = ["status", "current_query", "current_stage", "metrics"]
            missing = [f for f in required_fields if f not in data]

            if missing:
                self.print_test("Orchestrator Endpoint", "FAIL", f"Missing fields: {missing}")
                return {}

            self.print_test("Orchestrator Endpoint", "PASS", "All required fields present")

            # Check metrics
            metrics = data.get("metrics", {})
            self.print_test("Average Latency Metric", "PASS" if "avg_latency_ms" in metrics else "FAIL")
            self.print_test("Queries Per Second Metric", "PASS" if "queries_per_second" in metrics else "FAIL")
            self.print_test("Bottleneck Detection", "PASS" if "bottleneck_stage" in metrics else "FAIL")

            return data

        except Exception as e:
            self.print_test("Orchestrator Endpoint", "FAIL", f"Error: {e}")
            return {}

    def test_query_execution(self) -> Dict:
        """Test 3: Query execution and tracking."""
        self.print_header("Test 3: Query Execution & Tracking")

        test_query = {
            "text": "What is Thompson Sampling?",
            "mode": "direct",
            "max_steps": 5
        }

        try:
            # Execute query
            start = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json=test_query,
                timeout=30
            )
            latency = (time.time() - start) * 1000

            if response.status_code != 200:
                self.print_test("Query Execution", "FAIL", f"HTTP {response.status_code}")
                return {}

            data = response.json()

            # Check response fields
            self.print_test("Query Response", "PASS", f"Latency: {latency:.1f}ms")

            if "response" in data:
                self.print_test("Response Field", "PASS", f"Length: {len(data['response'])} chars")
            else:
                self.print_test("Response Field", "FAIL", "Missing response field")

            if "confidence" in data:
                conf = data["confidence"]
                self.print_test("Confidence Score", "PASS", f"Confidence: {conf:.2f}")
            else:
                self.print_test("Confidence Score", "FAIL", "Missing confidence field")

            # Check orchestrator tracking
            time.sleep(0.5)  # Wait for state update
            orch_data = requests.get(f"{self.base_url}/monitor/orchestrator").json()

            if orch_data.get("recent_traces"):
                self.print_test("Pipeline Trace Created", "PASS",
                              f"Traces: {len(orch_data['recent_traces'])}")
            else:
                self.print_test("Pipeline Trace Created", "WARN", "No traces found")

            return data

        except Exception as e:
            self.print_test("Query Execution", "FAIL", f"Error: {e}")
            return {}

    def test_learning_status(self) -> Dict:
        """Test 4: Learning system status (for Policy Monitor)."""
        self.print_header("Test 4: Learning System Status")

        try:
            response = requests.get(f"{self.base_url}/learning/status", timeout=5)

            if response.status_code != 200:
                self.print_test("Learning Status Endpoint", "FAIL", f"HTTP {response.status_code}")
                return {}

            data = response.json()

            # Check Thompson Sampling data
            if "thompson_sampling_stats" in data:
                ts_stats = data["thompson_sampling_stats"]
                if "arms" in ts_stats and ts_stats["arms"]:
                    arm_count = len(ts_stats["arms"])
                    self.print_test("Thompson Sampling Data", "PASS", f"{arm_count} arms tracked")

                    # Check arm structure
                    first_arm = list(ts_stats["arms"].values())[0]
                    required = ["alpha", "beta", "expected_reward"]
                    if all(k in first_arm for k in required):
                        self.print_test("Thompson Sampling Structure", "PASS")
                    else:
                        self.print_test("Thompson Sampling Structure", "FAIL", "Missing fields")
                else:
                    self.print_test("Thompson Sampling Data", "WARN", "No arms data (need queries)")
            else:
                self.print_test("Thompson Sampling Data", "FAIL", "Missing thompson_sampling_stats")

            # Check policy weights
            if "policy_weights" in data:
                weights = data["policy_weights"]
                modes = ["BARE", "FAST", "FUSED"]
                if all(m in weights for m in modes):
                    total = sum(weights[m] for m in modes)
                    self.print_test("Policy Weights", "PASS", f"Total: {total:.2f}")
                else:
                    self.print_test("Policy Weights", "FAIL", "Missing weight modes")
            else:
                self.print_test("Policy Weights", "FAIL", "Missing policy_weights")

            return data

        except Exception as e:
            self.print_test("Learning Status", "FAIL", f"Error: {e}")
            return {}

    def test_multiple_queries(self, count: int = 5) -> List[Dict]:
        """Test 5: Multiple queries for policy evolution."""
        self.print_header(f"Test 5: Multiple Queries (n={count})")

        queries = [
            {"text": "What is reinforcement learning?", "mode": "direct"},
            {"text": "Explain Bayesian optimization", "mode": "verify"},
            {"text": "Compare exploration strategies", "mode": "research", "max_steps": 3},
            {"text": "Define neural networks", "mode": "direct"},
            {"text": "What are embeddings?", "mode": "direct"},
        ]

        results = []
        latencies = []

        for i, query_data in enumerate(queries[:count]):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json=query_data,
                    timeout=30
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                if response.status_code == 200:
                    data = response.json()
                    results.append(data)
                    status = "PASS"
                else:
                    status = "FAIL"

                self.print_test(f"Query {i+1}/{count} ({query_data['mode']})",
                              status, f"{latency:.1f}ms")

                time.sleep(0.5)  # Small delay between queries

            except Exception as e:
                self.print_test(f"Query {i+1}/{count}", "FAIL", f"Error: {e}")

        # Summary
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            print(f"\n  Latency: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms")

        return results

    def test_performance_metrics(self):
        """Test 6: Performance benchmarks."""
        self.print_header("Test 6: Performance Benchmarks")

        # Make 10 queries and measure performance
        latencies = []
        confidences = []

        print("  Running 10 queries for benchmark...")
        for i in range(10):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={
                        "text": f"Test query {i+1}",
                        "mode": "direct",
                        "max_steps": 3
                    },
                    timeout=30
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                if response.status_code == 200:
                    data = response.json()
                    if "confidence" in data:
                        confidences.append(data["confidence"])

                time.sleep(0.3)

            except Exception as e:
                print(f"  Query {i+1} failed: {e}")

        # Calculate metrics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

            self.print_test("Average Latency",
                          "PASS" if avg_latency < 1000 else "WARN",
                          f"{avg_latency:.1f}ms (target: <1000ms)")

            self.print_test("P95 Latency",
                          "PASS" if p95_latency < 2000 else "WARN",
                          f"{p95_latency:.1f}ms (target: <2000ms)")

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            self.print_test("Average Confidence",
                          "PASS" if avg_conf > 0.5 else "WARN",
                          f"{avg_conf:.2f} (target: >0.5)")

    def test_bottleneck_detection(self):
        """Test 7: Bottleneck detection."""
        self.print_header("Test 7: Bottleneck Detection")

        # Make a complex query that might trigger bottleneck detection
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={
                    "text": "Explain the full 9-step weaving cycle in detail with examples",
                    "mode": "research",
                    "max_steps": 10
                },
                timeout=60
            )

            if response.status_code != 200:
                self.print_test("Complex Query", "FAIL", f"HTTP {response.status_code}")
                return

            latency = response.json().get("latency_ms", 0)
            self.print_test("Complex Query", "PASS", f"Latency: {latency:.1f}ms")

            # Check bottleneck detection
            time.sleep(0.5)
            orch_data = requests.get(f"{self.base_url}/monitor/orchestrator").json()

            bottleneck = orch_data.get("metrics", {}).get("bottleneck_stage")
            if bottleneck:
                self.print_test("Bottleneck Detection", "PASS", f"Detected: {bottleneck}")
            else:
                self.print_test("Bottleneck Detection", "WARN", "No bottleneck detected")

        except Exception as e:
            self.print_test("Bottleneck Detection", "FAIL", f"Error: {e}")

    def test_data_persistence(self):
        """Test 8: Data persistence across requests."""
        self.print_header("Test 8: Data Persistence")

        try:
            # Get initial state
            orch1 = requests.get(f"{self.base_url}/monitor/orchestrator").json()
            traces1 = len(orch1.get("recent_traces", []))

            # Make a query
            requests.post(
                f"{self.base_url}/query",
                json={"text": "Persistence test", "mode": "direct"},
                timeout=30
            )

            time.sleep(0.5)

            # Get state again
            orch2 = requests.get(f"{self.base_url}/monitor/orchestrator").json()
            traces2 = len(orch2.get("recent_traces", []))

            if traces2 > traces1:
                self.print_test("Trace Persistence", "PASS",
                              f"Traces: {traces1} → {traces2}")
            else:
                self.print_test("Trace Persistence", "FAIL",
                              "Trace count did not increase")

        except Exception as e:
            self.print_test("Data Persistence", "FAIL", f"Error: {e}")

    def generate_report(self):
        """Generate final test report."""
        self.print_header("Test Summary")

        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total * 100) if total > 0 else 0

        print(f"Total Tests: {total}")
        print(f"{Colors.OKGREEN}Passed: {self.tests_passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.tests_failed}{Colors.ENDC}")
        print(f"Pass Rate: {pass_rate:.1f}%\n")

        # Save results to JSON
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "pass_rate": pass_rate
            },
            "results": self.test_results
        }

        report_path = "test_phase3_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Detailed results saved to: {report_path}\n")

        # Return exit code
        return 0 if self.tests_failed == 0 else 1

    def run_all_tests(self):
        """Run complete test suite."""
        print(f"{Colors.BOLD}{Colors.HEADER}")
        print("=" * 80)
        print("Phase 3 Automated Test Suite".center(80))
        print("Enhanced Monitoring Dashboards".center(80))
        print("=" * 80)
        print(f"{Colors.ENDC}")

        # Check server first
        if not self.check_server_health():
            print(f"\n{Colors.FAIL}Server not available. Please start the server first.{Colors.ENDC}")
            print(f"Run: PYTHONPATH=. uvicorn HoloLoom.server.unified_server:app --reload --port 8000\n")
            return 1

        # Run test suite
        self.test_orchestrator_endpoint()
        self.test_query_execution()
        self.test_learning_status()
        self.test_multiple_queries(count=5)
        self.test_performance_metrics()
        self.test_bottleneck_detection()
        self.test_data_persistence()

        # Generate report
        return self.generate_report()


def main():
    """Main entry point."""
    tester = Phase3Tester(base_url="http://localhost:8000")
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
